import os

# This script updates all Python files in the codebase to use the new import for BaseSettings from pydantic-settings.
# It replaces: from pydantic_settings import BaseSettings\nfrom pydantic import Field
# With: from pydantic_settings import BaseSettings\nfrom pydantic import Field


def update_imports(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                new_content = content.replace(
                    "from pydantic_settings import BaseSettings\nfrom pydantic import Field",
                    "from pydantic_settings import BaseSettings\nfrom pydantic import Field",
                )
                if new_content != content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated: {file_path}")


if __name__ == "__main__":
    update_imports(os.path.dirname(os.path.abspath(__file__)))
    print("Done.")
