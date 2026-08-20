"""MapEval-API's five baseline tools, served by Kakao instead of the MapQaTor backend.

Ported from `MapEval/MapEval-API@35d481a:FormattedTools.py`. What is kept verbatim is
everything the baseline is measured on: the five tool classes, their names, descriptions and
`args_schema` fields, the place-id threading between them, and the four `*_to_context`
formatters that decide what an observation actually says. Upstream's evaluator instantiates
exactly `PlaceSearchTool, PlaceDetailsTool, NearbyPlacesTool, TravelTimeTool, DirectionsTool`
(`Evaluator2.py`, line 33) and this module still defines exactly those.

What changed is where an observation comes from. Upstream every `_run` was an HTTP GET to
`http://localhost:5000/api`, the MapQaTor backend, which proxied Google Maps. Those calls are
now `KakaoMapsClient` calls against Kakao Local and Kakao Mobility — the same client the
vendored Spatial-Agent reads, so neither architecture sees evidence the other cannot.

Three upstream behaviours could not survive the provider change, and each is a Kakao fact
rather than a choice:

- `place_to_context` prints a rating, a price level and opening hours when the place carries
  them. Kakao Local publishes none of the three, so those lines are simply absent — the
  formatter is unchanged and its conditions are false.
- `TravelTimeTool` and `DirectionsTool` take a `travelMode`. Kakao Mobility routes cars only,
  so a walking, cycling or transit request is refused in words rather than answered with a
  driving route.
- Upstream slept 30 seconds after every successful call, to stay inside the hosted backend's
  rate limit. Against Kakao with a local cache there is no such limit, and 30 s a call would
  add hours to a run without changing an answer. It is now `MAPEVAL_TOOL_SLEEP_SECONDS`,
  default 0.
"""

import os
import threading
import traceback
from typing import Optional, Type

import inflect
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.kakao_maps import KakaoAuthError, KakaoMapsClient, format_distance, format_duration

# Create an inflect engine
p = inflect.engine()

# Upstream read `types.json`, Google's place-type list, to decide whether a NearbyPlaces
# request was a `type` or a `keyword`. Kakao's own vocabulary answers the same question.
from src.kakao_maps import TYPE_VOCABULARY as types  # noqa: E402

# How long to sleep after a successful tool call. Upstream hard-coded 30 seconds for the
# hosted MapQaTor backend's rate limit; Kakao served directly needs none.
TOOL_SLEEP_SECONDS = float(os.getenv("MAPEVAL_TOOL_SLEEP_SECONDS", "0") or 0)

# The Evaluator runs one worker thread per concurrent question and each worker owns its own
# client, so that per-question API-call counters cannot cross-write. A tool is a pydantic
# model whose fields are part of the schema the agent sees, so the client is held beside the
# tools rather than on them.
_local = threading.local()


def set_client(client: Optional[KakaoMapsClient]) -> None:
    """Bind the Kakao client this thread's tools read."""

    _local.client = client


def get_client() -> KakaoMapsClient:
    """The Kakao client bound to this thread, constructing a default one if none is."""

    client = getattr(_local, "client", None)
    if client is None:
        client = KakaoMapsClient()
        _local.client = client
    return client


def _sleep():
    if TOOL_SLEEP_SECONDS > 0:
        import time

        time.sleep(TOOL_SLEEP_SECONDS)


def _driving_only(travelMode: str) -> Optional[str]:
    """Upstream's four travel modes against a provider that serves one.

    Answering a walking question with a driving route would answer a different question, so
    the refusal is explicit and names what is available.
    """

    if str(travelMode).lower() in {"driving", "car"}:
        return None
    return (
        f"Travel mode '{travelMode}' is not available from this map provider. "
        "Kakao Mobility serves driving routes only. Use travelMode='driving'."
    )


class TravelTime(BaseModel):
    originId: str = Field(description="Place Id of Origin")
    destinationId: str = Field(description="Place Id of Destination")
    travelMode: str = Field(description="Mode of transportation (driving, walking, bicycling, transit)")

class TravelTimeTool(BaseTool):
    name: str = "TravelTime"
    description: str = "Estimate the travel time between two places."
    args_schema: Type[BaseModel] = TravelTime
    handle_tool_error: bool  = True

    def _run(self, originId: str, destinationId: str, travelMode: str) -> str:
        refusal = _driving_only(travelMode)
        if refusal:
            return refusal

        client = get_client()
        try:
            matrix = client.get_distance_matrix([originId], [destinationId], mode="driving")
            element = (matrix or {}).get("rows", [{}])[0].get("elements", [{}])[0]

            if element.get("status") == "OK":
                _sleep()
                duration = element["duration"]["text"]
                distance = element["distance"]["text"]
                return f"Travel Time by car is {duration} ({distance})."
            return "No route found. Please check the place ids and try again."
        except KakaoAuthError:
            raise
        except Exception:
            traceback.print_exc()
            return "No route found. Please check the place ids and try again."

    def _arun(self, originId: str, destinationId: str, travelMode: str):
        raise NotImplementedError("This tool does not support async")


def place_to_context(place):
    text = ""

    lat = place['geometry']['location']['lat']() if callable(place['geometry']['location']['lat']) else place['geometry']['location']['lat']
    lng = place['geometry']['location']['lng']() if callable(place['geometry']['location']['lng']) else place['geometry']['location']['lng']
    text += f"- Location: {place.get('formatted_address', '')}{' (' + str(lat) + ', ' + str(lng) + ')' }.\n"

    if place.get("phone_number"):
        text += f"- Phone Number: {place.get('phone_number', '')}.\n"

    try:
        if place.get("opening_hours") and place["opening_hours"].get("weekday_text"):
            text += f"- Open: {', '.join(place['opening_hours']['weekday_text'])}.\n"
    except:
        pass

    if place.get("rating"):
        text += f"- Rating: {place.get('rating', '')}. ({place.get('user_ratings_total', '0')} ratings).\n"

    if place.get("price_level"):
        price_map = ["Free", "Inexpensive", "Moderate", "Expensive", "Very Expensive"]
        price_level = price_map[place.get('price_level', 0)]
        text += f"- Price Level: {price_level}.\n"

    if place.get("delivery"):
        text += "- Delivery Available.\n" if place.get('delivery') else "- Delivery Not Available.\n"

    if place.get("dine_in"):
        text += "- Dine In Available.\n" if place.get('dine_in') else "- Dine In Not Available.\n"

    if place.get("reservable"):
        text += "- Reservable.\n" if place.get('reservable') else "- Not Reservable.\n"

    if place.get("serves_breakfast"):
        text += "- Serves Breakfast.\n" if place.get('serves_breakfast') else "- Does Not Serve Breakfast.\n"

    if place.get("serves_lunch"):
        text += "- Serves Lunch.\n" if place.get('serves_lunch') else "- Does Not Serve Lunch.\n"

    if place.get("serves_dinner"):
        text += "- Serves Dinner.\n" if place.get('serves_dinner') else "- Does Not Serve Dinner.\n"

    if place.get("takeout"):
        text += "- Takeout Available.\n" if place.get('takeout') else "- Takeout Not Available.\n"

    if place.get("wheelchair_accessible_entrance"):
        text += "- Wheelchair Accessible Entrance.\n" if place.get('wheelchair_accessible_entrance') else "- Not Wheelchair Accessible Entrance.\n"

    return text


def _as_google_details(details):
    """One `KakaoMapsClient.get_place_details` result in the shape `place_to_context` reads.

    The formatter is upstream's, unchanged, so what it is handed has to be Google's Place
    Details shape: nested `geometry.location`, a `phone_number`, and the attribute keys it
    tests for. The keys Kakao cannot fill are left out entirely rather than defaulted, which
    is what makes the formatter's own conditions skip them.
    """

    return {
        "name": details["name"],
        "place_id": details["place_id"],
        "geometry": {"location": {"lat": details["lat"], "lng": details["lng"]}},
        "formatted_address": details.get("formatted_address", ""),
        "phone_number": details.get("formatted_phone_number"),
        "types": details.get("types", []),
    }


class PlaceDetails(BaseModel):
    placeId: str = Field(description="Place Id of the location")


class PlaceDetailsTool(BaseTool):
    name: str = "PlaceDetails"
    description: str = "Get details for a given place ID."
    args_schema: Type[BaseModel] = PlaceDetails
    handle_tool_error: bool  = True

    def _run(self, placeId: str) -> str:
        client = get_client()
        details = client.get_place_details(placeId)
        if details is None:
            return "Incorrect Place ID. Please use correct place id."
        _sleep()
        try:
            return place_to_context(_as_google_details(details))
        except KakaoAuthError:
            raise
        except Exception as e:
            print(f"Failed to retrieve data: {e}")
            traceback.print_exc()
            return "Incorrect Place ID. Please use correct place id."


class PlaceSearch(BaseModel):
    placeName: str = Field(description="Name and address of the place")

class PlaceSearchTool(BaseTool):
    name: str = "PlaceSearch"
    description: str = "Get place ID for a given location."
    args_schema: Type[BaseModel] = PlaceSearch
    handle_tool_error: bool  = True

    def _run(self, placeName: str) -> str:
        # Upstream is `return data['results'][0]['place_id']` — an id and nothing else, so a
        # baseline that wants coordinates pays a second round trip through PlaceDetails.
        # That asymmetry is the paper's own arrangement and is preserved here.
        client = get_client()
        try:
            results = client.text_search(placeName)
            if results:
                _sleep()
                return results[0]["place_id"]
            return "Incorrect place name. Please use the same name as in the question."
        except KakaoAuthError:
            raise
        except Exception as e:
            print(f"Failed to retrieve data: {e}")
            traceback.print_exc()
            return "Incorrect place name. Please use the same name as in the question."


def directions_to_context(directions, mode):
    mode = mode.lower()
    text = ""
    if mode == "transit":
        text += f"There are {len(directions)} routes by public transport. They are:\n"
    elif mode == "driving":
        text +=  f"There are {len(directions)}  routes by car. They are:\n"
    elif mode == "bicycling":
        text +=  f"There are {len(directions)} routes by cycle. They are:\n"
    elif mode == "walking":
        text += f"There are {len(directions)} routes on foot. They are:\n"

    for index, route in enumerate(directions):
        text +=  f"{index + 1}. Via {route['summary']} | {route['legs'][0]['duration']['text']} | {route['legs'][0]['distance']['text']}\n"
        for step_index, step in enumerate(route['legs'][0]['steps']):
            text += f" - {step['html_instructions']}\n"
    return text


def _as_google_routes(result):
    """`KakaoMapsClient.get_directions` output in the shape `directions_to_context` reads.

    Upstream's formatter walks `route['legs'][0]`, because Google splits a route into legs at
    each waypoint. These calls have no waypoints, so every route is one leg, and the wrapping
    is what keeps the formatter unchanged.
    """

    routes = []
    for route in result.get("routes", []):
        routes.append(
            {
                "summary": route["summary"],
                "legs": [
                    {
                        "duration": {"text": route["duration_text"], "value": route["duration"]},
                        "distance": {"text": route["distance_text"], "value": route["distance"]},
                        "steps": route["steps"],
                    }
                ],
            }
        )
    return routes


class Directions(BaseModel):
    originId: str = Field(description="Place Id of Origin")
    destinationId: str = Field(description="Place Id of Destination")
    travelMode: str = Field(description="Mode of transportation (driving, walking, bicycling, transit)")

class DirectionsTool(BaseTool):
    name: str = "Directions"
    description: str = "Get directions/routes between two places."
    args_schema: Type[BaseModel] = Directions
    handle_tool_error: bool  = True

    def _run(self, originId: str, destinationId: str, travelMode: str) -> str:
        refusal = _driving_only(travelMode)
        if refusal:
            return refusal

        client = get_client()
        try:
            # Upstream's DirectionsTool prints every step on every call, and asks the backend
            # for alternatives, so a "which route" question has routes to compare.
            result = client.get_directions(originId, destinationId, mode="driving", alternatives=True)
            if not result or not result.get("routes"):
                return "No route found. Please check the place ids and try again."
            _sleep()
            return directions_to_context(_as_google_routes(result), travelMode)
        except KakaoAuthError:
            raise
        except Exception as e:
            print(f"Failed to retrieve data: {e}")
            traceback.print_exc()
            return "No route found. Please check the place ids and try again."

    def _arun(self, originId: str, destinationId: str, travelMode: str):
        raise NotImplementedError("This tool does not support async")

def convert_from_snake(snake_str):
    """
    Convert a snake_case string to a more readable format.
    """
    components = snake_str.split('_')
    return ' '.join(x.capitalize() for x in components)

def nearby_to_context(places, type, rankby, radius):
    text = ''
    text += f"Nearby {p.plural(convert_from_snake(type))} are ({'in ' + str(radius) + ' m radius' if rankby == 'prominence' else 'sorted by distance in ascending order'}):\n"
    counter = 1
    for near_place in places:
        text += f"{counter}. {near_place.get('name', 'Unknown')} ({near_place.get('place_id')})\n"

        if near_place.get('vicinity'):
            text += f"   - Address: {near_place.get('vicinity', '')}.\n"

        if near_place.get('rating'):
            text += f"   - Rating: {near_place.get('rating', '')}. ({near_place.get('user_ratings_total', '0')} ratings).\n"

        price_map = [
            "Free",
            "Inexpensive",
            "Moderate",
            "Expensive",
            "Very Expensive",
        ]
        if near_place.get('price_level') is not None:
            text += f"   - Price Level: {price_map[near_place.get('price_level', 0)]}.\n"

        try:
            if near_place.get('opening_hours') and near_place['opening_hours'].get('weekday_text'):
                text += f"   - Open: {', '.join(near_place['opening_hours']['weekday_text'])}.\n"
        except:
            pass

        counter += 1
    return text


class NearbyPlaces(BaseModel):
    placeId: str = Field(description="The id of the place around which to retrieve nearby places.")
    type: str = Field(description="Type of place (e.g., restaurant, hospital, etc). Restricts the results to places matching the specified type.")
    rankby: str = Field(default='distance', description="Specifies the order in which places are listed. Possible values are: (1. prominence (default): This option sorts results based on their importance. When prominence is specified, the radius parameter is required. 2. distance: This option sorts places in ascending order by their distance from the specified location. When distance is specified, radius is disallowed. In case you are not concerned about the radius, use rankby as distance.)")
    radius: Optional[int] = Field(default=None,description="Defines the distance (in meters) within which to return place results.")


class NearbyPlacesTool(BaseTool):
    name: str = "NearbyPlaces"
    description: str = "Get nearby places around a location."
    args_schema: Type[BaseModel] = NearbyPlaces
    handle_tool_error: bool = True

    def _run(self, placeId: str, type: str,  rankby: str = 'distance', radius: Optional[int] = None) -> str:
        if rankby == "distance" and radius is not None and radius > 0:
            return "When rankby is distance, radius is disallowed. If want to use rankby as distance, please set radius to 0. And if you want to use radius, please set rankby as prominence."

        client = get_client()
        try:
            anchor = client.get_place_details(placeId)
            if anchor is None:
                return "No places found. Please check the place id and other parameters and try again."

            # Upstream split the request the same way: a `type` when Google knew the token,
            # a `keyword` otherwise. Kakao's category codes stand in for Google's types.
            place_type = type if type in types else None
            keyword = type if type not in types else None

            # `rankby=distance` disallowed a radius upstream, which is Google's rule. Kakao
            # always takes a radius, so the unbounded ranking is asked for at Kakao's maximum
            # and still returned in distance order.
            search_radius = radius if (rankby == "prominence" and radius) else 20000

            results = client.nearby_search(
                location=(anchor["lat"], anchor["lng"]),
                radius=search_radius,
                place_type=place_type,
                keyword=keyword,
            )

            # A place is not among its own neighbours: the anchor stands at zero metres from
            # itself and would head every ranking it appears in.
            results = [place for place in results if place.get("place_id") != placeId]
            if not results:
                return "No places found. Please check the place id and other parameters and try again."
            _sleep()
            return nearby_to_context(results, type, rankby, radius)
        except KakaoAuthError:
            raise
        except Exception as e:
            print(f"Failed to retrieve data: {e}")
            traceback.print_exc()
            return "No places found. Please check the place id and other parameters and try again."

    def _arun(self, placeId: str, type: str,  rankby: str = 'distance', radius: Optional[int] = None):
        raise NotImplementedError("This tool does not support async")


__all__ = [
    "PlaceSearchTool",
    "PlaceDetailsTool",
    "NearbyPlacesTool",
    "TravelTimeTool",
    "DirectionsTool",
    "set_client",
    "get_client",
    "place_to_context",
    "directions_to_context",
    "nearby_to_context",
    "convert_from_snake",
    "format_distance",
    "format_duration",
]
