"""Test that optional dependencies use try/except when imported at module level.

The openbench registry imports all modules when running bench commands (list, eval, etc.),
so any module-level imports of optional dependencies will cause ImportErrors for users
who don't have those dependencies installed.

This prevents the warning:
    WARNING Unexpected exception loading entrypoints from 'openbench._registry':
    No module named '<dependency>'

Solution: Wrap module-level imports of optional dependencies in try/except blocks:
    try:
        import optional_package
    except ImportError:
        optional_package = None

This allows the module to load successfully, and you can check for the dependency
later when it's actually needed.

See: https://github.com/groq/openbench/pull/307
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest
import tomllib  # Python 3.11+


def normalize_package_to_import(package_name: str) -> List[str]:
    """Convert a package name to potential import names using common patterns.

    Returns a list of possible import names to check for.

    Examples:
        python-levenshtein -> [Levenshtein, python_levenshtein]
        google-api-python-client -> [googleapiclient, google, google_api_python_client]
        rank-bm25 -> [rank_bm25]
        langdetect -> [langdetect]
    """
    import_names = []

    # Common transformation: replace hyphens with underscores
    normalized = package_name.replace("-", "_")
    import_names.append(normalized)

    # For packages with "python-" prefix, try without prefix
    if package_name.startswith("python-"):
        without_prefix = package_name[7:]  # Remove "python-"
        # Capitalize first letter (common pattern)
        import_names.append(without_prefix.capitalize())
        # Also try lowercase with underscores
        import_names.append(without_prefix.replace("-", "_"))

    # For packages with multiple parts, try first part (e.g., google-genai -> google)
    if "-" in package_name:
        parts = package_name.split("-")
        import_names.append(parts[0])

        # For google-api-* patterns, try concatenated lowercase
        if len(parts) > 1:
            # Remove common suffixes like "python", "client"
            filtered_parts = [p for p in parts if p not in ("python", "client", "py")]
            if filtered_parts:
                import_names.append("".join(filtered_parts))

    # Always include the original package name
    import_names.append(package_name)

    return import_names


def get_transitive_dependencies(package_names: Set[str]) -> Set[str]:
    """Get all transitive dependencies of given packages using uv pip show.

    Args:
        package_names: Set of package names to check

    Returns:
        Set of all transitive dependency package names (lowercase)
    """
    import subprocess

    transitive_deps: Set[str] = set()

    for pkg in package_names:
        try:
            # Use uv pip show to get package dependencies
            result = subprocess.run(
                ["uv", "pip", "show", pkg],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                # Parse "Requires:" line
                for line in result.stdout.split("\n"):
                    if line.startswith("Requires:"):
                        # Extract comma-separated list of dependencies
                        deps_str = line.split(":", 1)[1].strip()
                        if deps_str and deps_str != "None":
                            deps = [d.strip().lower() for d in deps_str.split(",")]
                            transitive_deps.update(deps)
        except Exception:
            # If subprocess fails, skip this package
            continue

    return transitive_deps


def parse_optional_dependencies() -> Set[str]:
    """Parse pyproject.toml to extract optional dependency package names.

    Returns:
        Set of package names for optional dependencies (excluding 'dev' group,
        main dependencies, and transitive dependencies of main packages)
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    # Get main dependencies
    main_deps = pyproject.get("project", {}).get("dependencies", [])
    main_packages: Set[str] = set()
    for dep in main_deps:
        package_name = re.split(r"[>=<\[!]", dep)[0].strip().lower()
        main_packages.add(package_name)

    # Get transitive dependencies of main packages
    transitive_deps = get_transitive_dependencies(main_packages)

    # Combine main packages and their transitive dependencies
    all_main_deps = main_packages | transitive_deps

    dependency_groups = pyproject.get("dependency-groups", {})

    # Get all optional dependency package names (exclude 'dev' group and main dependencies)
    optional_packages: Set[str] = set()
    for group_name, deps in dependency_groups.items():
        if group_name == "dev":
            continue  # Skip dev dependencies

        for dep in deps:
            # Extract package name from version specifier
            # e.g., "langdetect>=1.0.9" -> "langdetect"
            package_name = re.split(r"[>=<\[!]", dep)[0].strip().lower()

            # Skip if this package is in main dependencies or their transitive deps
            if package_name in all_main_deps:
                continue

            optional_packages.add(package_name)

    return optional_packages


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to find module-level imports without try/except protection."""

    def __init__(self, optional_packages: Set[str]):
        # Build mapping from potential import names to package names
        self.import_to_package: Dict[str, str] = {}
        for pkg in optional_packages:
            for import_name in normalize_package_to_import(pkg):
                self.import_to_package[import_name.lower()] = pkg

        self.violations: List[Tuple[int, str, str]] = []  # (lineno, import_name, type)
        self.in_function = False
        self.in_class = False
        self.in_try_except = False
        self.in_type_checking = False

    def visit_If(self, node):
        """Track TYPE_CHECKING blocks (imports inside these are okay)."""
        # Check if this is: if TYPE_CHECKING:
        is_type_checking = False
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            is_type_checking = True

        if is_type_checking:
            old_in_type_checking = self.in_type_checking
            self.in_type_checking = True
            self.generic_visit(node)
            self.in_type_checking = old_in_type_checking
        else:
            self.generic_visit(node)

    def visit_Try(self, node):
        """Track when we're inside a try/except block."""
        old_in_try_except = self.in_try_except
        self.in_try_except = True
        # Only visit the try body, not the except handlers
        for child in node.body:
            self.visit(child)
        self.in_try_except = old_in_try_except
        # Visit except handlers and else/finally with normal context
        for handler in node.handlers:
            self.visit(handler)
        for child in node.orelse:
            self.visit(child)
        for child in node.finalbody:
            self.visit(child)

    def visit_FunctionDef(self, node):
        """Track when we're inside a function."""
        old_in_function = self.in_function
        self.in_function = True
        self.generic_visit(node)
        self.in_function = old_in_function

    def visit_AsyncFunctionDef(self, node):
        """Track when we're inside an async function."""
        old_in_function = self.in_function
        self.in_function = True
        self.generic_visit(node)
        self.in_function = old_in_function

    def visit_ClassDef(self, node):
        """Track when we're inside a class."""
        old_in_class = self.in_class
        self.in_class = True
        self.generic_visit(node)
        self.in_class = old_in_class

    def visit_Import(self, node):
        """Check 'import x' statements."""
        # Only flag imports at module level (not inside functions/classes/try-except/TYPE_CHECKING)
        if not self.in_function and not self.in_try_except and not self.in_type_checking:
            for alias in node.names:
                module = alias.name.split(".")[0].lower()  # Get base module name
                if module in self.import_to_package:
                    self.violations.append((node.lineno, alias.name, "import"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check 'from x import y' statements."""
        # Only flag imports at module level without try/except protection
        if not self.in_function and not self.in_try_except and not self.in_type_checking and node.module:
            module = node.module.split(".")[0].lower()  # Get base module name
            if module in self.import_to_package:
                imported_names = ", ".join(
                    alias.name if not alias.asname else f"{alias.name} as {alias.asname}"
                    for alias in node.names
                )
                self.violations.append(
                    (node.lineno, f"{node.module}.{imported_names}", "from")
                )
        self.generic_visit(node)


def find_openbench_imports(file_path: Path, src_dir: Path) -> Set[Path]:
    """Find all openbench modules imported by this file.

    Args:
        file_path: Path to Python file to analyze
        src_dir: Root directory of openbench package

    Returns:
        Set of file paths to openbench modules imported by this file
    """
    imported_files = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except (SyntaxError, FileNotFoundError):
        return imported_files

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # Only care about relative imports or openbench imports
            if node.module and (node.level > 0 or node.module.startswith("openbench")):
                # Convert import to file path
                if node.level > 0:
                    # Relative import: from .evals.docvqa import docvqa
                    # Compute relative path from current file
                    current_dir = file_path.parent
                    for _ in range(node.level - 1):
                        current_dir = current_dir.parent
                    module_parts = node.module.split(".") if node.module else []
                    module_path = current_dir.joinpath(*module_parts)
                else:
                    # Absolute openbench import: from openbench.evals.docvqa import docvqa
                    module_parts = node.module.split(".")
                    if module_parts[0] == "openbench":
                        module_parts = module_parts[1:]  # Remove 'openbench' prefix
                    module_path = src_dir / "openbench" / "/".join(module_parts)

                # Try as both file and package
                if module_path.with_suffix(".py").exists():
                    imported_files.add(module_path.with_suffix(".py"))
                elif (module_path / "__init__.py").exists():
                    imported_files.add(module_path / "__init__.py")

        elif isinstance(node, ast.Import):
            # Check for openbench imports
            for alias in node.names:
                if alias.name.startswith("openbench."):
                    module_parts = alias.name.split(".")[1:]  # Remove 'openbench' prefix
                    module_path = src_dir / "openbench" / "/".join(module_parts)

                    if module_path.with_suffix(".py").exists():
                        imported_files.add(module_path.with_suffix(".py"))
                    elif (module_path / "__init__.py").exists():
                        imported_files.add(module_path / "__init__.py")

    return imported_files


def trace_registry_imports(registry_path: Path, src_dir: Path) -> Set[Path]:
    """Trace all files transitively imported by the registry.

    Args:
        registry_path: Path to _registry.py file
        src_dir: Root directory of openbench package

    Returns:
        Set of all file paths that are transitively imported by the registry
    """
    visited = set()
    to_visit = {registry_path}
    all_files = {registry_path}

    while to_visit:
        current_file = to_visit.pop()
        if current_file in visited:
            continue

        visited.add(current_file)

        # Find imports in this file
        imports = find_openbench_imports(current_file, src_dir)

        # Add new imports to the set
        for imported_file in imports:
            if imported_file not in visited:
                to_visit.add(imported_file)
                all_files.add(imported_file)

    return all_files


def find_global_optional_imports(
    file_path: Path, optional_packages: Set[str]
) -> List[Tuple[int, str, str]]:
    """Find global (module-level) imports of optional dependencies.

    Args:
        file_path: Path to Python file to analyze
        optional_packages: Set of optional dependency package names to check for

    Returns:
        List of (line_number, import_name, import_type) tuples for violations
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except SyntaxError:
        # Skip files with syntax errors
        return []

    visitor = ImportVisitor(optional_packages)
    visitor.visit(tree)
    return visitor.violations


def test_no_global_optional_dependency_imports():
    """Test that optional dependencies use try/except when imported at module level.

    This test traces the import chain starting from _registry.py (the inspect_ai entrypoint)
    and only checks files that are actually imported by the registry. This prevents false
    positives for files that are never loaded at import time.

    Dynamically reads pyproject.toml to find all optional dependencies in dependency-groups
    (excluding 'dev'), then verifies that any module-level imports of these packages are
    wrapped in try/except blocks.

    This prevents the registry loading error:
        WARNING Unexpected exception loading entrypoints from 'openbench._registry':
        No module named '<dependency>'
    """
    # Dynamically parse optional dependencies from pyproject.toml
    optional_packages = parse_optional_dependencies()

    # Find the registry file
    src_dir = Path(__file__).parent.parent / "src"
    registry_path = src_dir / "openbench" / "_registry.py"

    # Trace all files imported by the registry
    registry_files = trace_registry_imports(registry_path, src_dir)

    # Check only files that are actually imported by the registry
    violations = []
    for py_file in registry_files:
        file_violations = find_global_optional_imports(py_file, optional_packages)
        if file_violations:
            rel_path = py_file.relative_to(src_dir)
            violations.extend([(str(rel_path), line, name, type_) for line, name, type_ in file_violations])

    # Report violations
    if violations:
        error_msg = [
            "\n❌ Found module-level imports of optional dependencies without try/except:",
            "\nThe registry imports all modules, so optional dependencies MUST be wrapped",
            "in try/except blocks to prevent ImportErrors.\n",
        ]
        for file_path, line, import_name, import_type in violations:
            error_msg.append(f"  {file_path}:{line} - {import_type} {import_name}")

        error_msg.append(
            "\n💡 Fix: Wrap module-level imports in try/except blocks:"
        )
        error_msg.append("    try:")
        error_msg.append("        import optional_package  # type: ignore[import-not-found]")
        error_msg.append("    except ImportError:")
        error_msg.append("        optional_package = None  # type: ignore[assignment]")
        error_msg.append("")
        error_msg.append("    # Then check before use:")
        error_msg.append("    if optional_package is None:")
        error_msg.append('        raise RuntimeError("Please install: uv sync --group <group_name>")')
        error_msg.append("\nSee: https://github.com/groq/openbench/pull/307")

        pytest.fail("\n".join(error_msg), pytrace=False)


if __name__ == "__main__":
    # Allow running the test directly for debugging
    test_no_global_optional_dependency_imports()
    print("✅ No global imports of optional dependencies found!")
