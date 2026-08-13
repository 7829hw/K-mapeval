from scripts.common import parser, run


def main() -> None:
    args = parser("Run the Kakao-ported Spatial-Agent").parse_args()
    run("spatial_agent", args)


if __name__ == "__main__":
    main()

