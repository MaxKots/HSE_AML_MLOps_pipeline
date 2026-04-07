from dataclasses import dataclass, field


@dataclass
class DataValidationReport:
    is_valid: bool
    row_count: int
    column_count: int
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    columns_with_nulls: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    message: str = ""
