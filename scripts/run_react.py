from scripts.common import parser, run


def main() -> None:
    args = parser("Run the MapEval-style ReAct baseline").parse_args()
    run("react", args)


if __name__ == "__main__":
    main()

