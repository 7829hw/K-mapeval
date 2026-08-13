from __future__ import annotations

from scripts.common import parser, run


def main() -> None:
    args = parser("Run both agents with identical settings").parse_args()
    react = run("react", args)
    spatial = run("spatial_agent", args)
    print(
        f"ReAct accuracy={react['accuracy']:.3f} | "
        f"Spatial-Agent accuracy={spatial['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()

