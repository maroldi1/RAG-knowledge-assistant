"""Run ingestion or query scripts inside an E2B sandbox."""

import argparse
import logging
import os
import sys

sys.path.insert(0, ".")

from e2b import Sandbox

from src import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _upload_tree(sandbox: Sandbox, local_dir: str, remote_dir: str) -> None:
    """Recursively upload a local directory into the sandbox."""
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, local_dir)
            remote_path = f"{remote_dir}/{rel}"
            with open(local_path, "rb") as f:
                sandbox.files.write(remote_path, f.read())


def main() -> None:
    """Create a sandbox and run the requested script."""
    parser = argparse.ArgumentParser(description="Run scripts in E2B sandbox")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["ingest", "query"],
        help="Which script to run",
    )
    parser.add_argument("--docs-dir", help="Docs directory (ingest mode)")
    parser.add_argument("--question", help="Question to ask (query mode)")
    parser.add_argument("--reset", action="store_true", help="Reset collection first")
    args = parser.parse_args()

    sandbox = Sandbox(template="base", api_key=config.E2B_API_KEY)
    try:
        logger.info("Sandbox created: %s", sandbox.sandbox_id)

        # Upload project files
        _upload_tree(sandbox, os.path.join(_PROJECT_ROOT, "src"), "/home/user/src")
        for script in ("ingest.py", "query.py"):
            local = os.path.join(_PROJECT_ROOT, "scripts", script)
            with open(local, "rb") as f:
                sandbox.files.write(f"/home/user/scripts/{script}", f.read())

        # Upload requirements and .env
        for filename in ("requirements.txt", ".env"):
            local = os.path.join(_PROJECT_ROOT, filename)
            if os.path.exists(local):
                with open(local, "rb") as f:
                    sandbox.files.write(f"/home/user/{filename}", f.read())

        # Install dependencies
        logger.info("Installing dependencies...")
        result = sandbox.commands.run("pip install -r /home/user/requirements.txt")
        if result.exit_code != 0:
            logger.error("pip install failed:\n%s", result.stderr)
            sys.exit(1)

        # Upload docs if ingesting
        if args.mode == "ingest":
            if not args.docs_dir:
                logger.error("--docs-dir is required for ingest mode")
                sys.exit(1)
            _upload_tree(sandbox, args.docs_dir, "/home/user/docs")

        # Build command
        if args.mode == "ingest":
            cmd = "cd /home/user && python scripts/ingest.py --docs-dir ./docs"
            if args.reset:
                cmd += " --reset"
        else:
            if not args.question:
                logger.error("--question is required for query mode in sandbox")
                sys.exit(1)
            cmd = (
                f'cd /home/user && python scripts/query.py '
                f'--question "{args.question}"'
            )

        # Run and stream output
        logger.info("Running: %s", cmd)
        result = sandbox.commands.run(
            cmd,
            on_stdout=lambda data: print(data if isinstance(data, str) else data.line),
            on_stderr=lambda data: print(data if isinstance(data, str) else data.line, file=sys.stderr),
        )

        if result.exit_code != 0:
            logger.error("Script exited with code %d", result.exit_code)
            sys.exit(result.exit_code)

    finally:
        sandbox.kill()
        logger.info("Sandbox killed")


if __name__ == "__main__":
    main()
