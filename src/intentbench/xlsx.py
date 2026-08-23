"""Small deterministic XLSX value reader for hash-bound human workbooks."""

from __future__ import annotations

import posixpath
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree

SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REFERENCE = re.compile(r"^([A-Z]{1,3})([1-9][0-9]*)$")


class XlsxReadError(ValueError):
    """An XLSX archive cannot be read as the expected bounded table."""


def xlsx_column_number(name: str) -> int:
    value = 0
    for character in name:
        value = value * 26 + ord(character) - ord("A") + 1
    return value


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(path))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{SPREADSHEET_NS}}}t"))
        for item in root.findall(f"{{{SPREADSHEET_NS}}}si")
    ]


def _sheet_paths(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook_root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships_root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships_root.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    paths: dict[str, str] = {}
    for sheet in workbook_root.findall(f"{{{SPREADSHEET_NS}}}sheets/{{{SPREADSHEET_NS}}}sheet"):
        relationship_id = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
        target = targets.get(relationship_id)
        if target is None:
            raise XlsxReadError(f"workbook sheet relationship missing: {relationship_id}")
        normalized = posixpath.normpath(
            target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
        )
        if not normalized.startswith("xl/"):
            raise XlsxReadError("workbook sheet relationship escapes xl directory")
        paths[sheet.attrib["name"]] = normalized
    return paths


def _cell_value(
    cell: ElementTree.Element, shared_strings: Sequence[str]
) -> str | int | float | bool | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        value = "".join(node.text or "" for node in cell.iter(f"{{{SPREADSHEET_NS}}}t"))
        return value or None
    value_node = cell.find(f"{{{SPREADSHEET_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw_value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (IndexError, ValueError) as exc:
            raise XlsxReadError("workbook contains an invalid shared-string index") from exc
    if cell_type in {"str", "e", "d"}:
        return raw_value
    if cell_type == "b":
        return raw_value == "1"
    try:
        if re.fullmatch(r"-?[0-9]+", raw_value):
            return int(raw_value)
        return float(raw_value)
    except ValueError:
        return raw_value


def load_xlsx_cells(path: Path, required_sheets: Sequence[str]) -> dict[str, dict[str, object]]:
    """Read cached values from named XLSX sheets without executing formulas."""

    if not zipfile.is_zipfile(path):
        raise XlsxReadError(f"not a valid .xlsx archive: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _shared_strings(archive)
            sheet_paths = _sheet_paths(archive)
            missing = set(required_sheets) - set(sheet_paths)
            if missing:
                raise XlsxReadError(f"workbook sheets missing: {sorted(missing)}")
            result: dict[str, dict[str, object]] = {}
            for sheet_name in required_sheets:
                root = ElementTree.fromstring(archive.read(sheet_paths[sheet_name]))
                cells: dict[str, object] = {}
                for cell in root.iter(f"{{{SPREADSHEET_NS}}}c"):
                    reference = cell.attrib.get("r")
                    if reference is None or CELL_REFERENCE.fullmatch(reference) is None:
                        raise XlsxReadError(f"{sheet_name}: invalid or missing cell reference")
                    cells[reference] = _cell_value(cell, shared_strings)
                result[sheet_name] = cells
            return result
    except (KeyError, ElementTree.ParseError, zipfile.BadZipFile) as exc:
        raise XlsxReadError(f"invalid .xlsx workbook structure: {path}") from exc
