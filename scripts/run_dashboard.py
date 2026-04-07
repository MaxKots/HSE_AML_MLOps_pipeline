import subprocess


def main() -> None:
    subprocess.run(
        [
            "streamlit",
            "run",
            "dashboard/app.py",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
