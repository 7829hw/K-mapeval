# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

The guidance itself lives in @AGENT.md — what this branch is, commands, layering, the rules that
break silently, and dataset conventions. It is the single source of truth shared with other
coding agents; make edits there, and keep this file a pointer.

One thing to know before touching anything: on this branch **both agents are upstream's code,
vendored unmodified, and the only thing swapped is the map API**. `src/spatial_agent/` and
`src/mapeval_api/Evaluator2.py` are not ours to edit. Every deviation from the two upstreams is
recorded in @docs/UPSTREAM_MAPPING.md; if a change you are about to make is not in there, either
it does not belong or the document needs a new line first.
