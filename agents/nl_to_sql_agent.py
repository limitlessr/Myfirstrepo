"""
NLToSQLAgent — translates natural-language questions into SQL queries.

Flow:
  1. Accept a natural-language question and an optional database schema.
  2. Ask Claude to generate a safe, read-only SQL query.
  3. Optionally execute the query against a SQLite database and return results.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from .base_agent import BaseAgent, AgentResult


_SYSTEM = """You are an expert SQL engineer. Your job is to translate a
natural-language question into a correct, read-only SQL SELECT query.

Rules:
- Output ONLY a JSON object — no prose, no markdown fences.
- The JSON must have exactly these keys:
    "sql"         : the SQL query string (SELECT only — never INSERT/UPDATE/DELETE/DROP)
    "explanation" : one sentence describing what the query does
    "params"      : a list of bind-parameter values in order ([] if none)
- Use standard ANSI SQL unless the dialect is specified.
- Prefer parameterised queries (? placeholders) to prevent injection.
- If the question cannot be answered from the schema, set "sql" to "" and
  explain why in "explanation".

Schema format you will receive:
  CREATE TABLE statements, one per table, separated by semicolons."""


_EXEC_SYSTEM = """You are an expert data analyst. The user ran a SQL query and
received tabular results. Summarise the key findings in 2-4 bullet points.
Be concise and precise. Reference specific numbers where useful."""


class NLToSQLAgent(BaseAgent):
    """Translates natural-language questions to SQL, with optional execution."""

    name = "NLToSQLAgent"
    description = "Converts natural-language questions into SQL queries using Claude"
    emoji = "🗄️"

    def process(
        self,
        *,
        question: str,
        schema: str = "",
        db_path: str | None = None,
        dialect: str = "sqlite",
        max_rows: int = 100,
        **_,
    ) -> AgentResult:
        """
        Args:
            question:  Natural-language question to answer.
            schema:    DDL string (CREATE TABLE …) describing the database.
                       Optional but strongly recommended for accuracy.
            db_path:   Path to a SQLite file. When provided, the generated
                       query is executed and results are returned.
            dialect:   SQL dialect hint passed to Claude (default: "sqlite").
            max_rows:  Maximum rows to return when executing (default: 100).
        """
        self._log(f"Question: '{question[:80]}'")

        # --- Build prompt -----------------------------------------------
        schema_section = (
            f"Database schema ({dialect}):\n{schema}\n\n" if schema else
            f"No schema provided. Assume a reasonable schema for {dialect}.\n\n"
        )
        user_content = f"{schema_section}Question: {question}"

        # --- Generate SQL -----------------------------------------------
        self._log("Generating SQL with Claude…")
        raw = self._call_claude(_SYSTEM, user_content, max_tokens=1024)

        try:
            payload = json.loads(raw.strip().strip("```json").strip("```").strip())
        except json.JSONDecodeError as exc:
            return AgentResult(
                agent=self.name,
                success=False,
                data=None,
                message=f"Claude returned non-JSON output: {exc}\n\nRaw: {raw[:300]}",
            )

        sql: str = payload.get("sql", "").strip()
        explanation: str = payload.get("explanation", "")
        params: list[Any] = payload.get("params", [])

        if not sql:
            return AgentResult(
                agent=self.name,
                success=True,
                data={"sql": "", "explanation": explanation, "params": [], "rows": None},
                message=explanation or "Could not generate SQL for this question.",
            )

        self._log(f"SQL generated: {sql[:120]}")

        # --- Execute (optional) -----------------------------------------
        rows: list[dict] | None = None
        columns: list[str] = []
        exec_summary: str = ""

        if db_path:
            rows, columns, exec_summary = self._execute(
                db_path=db_path, sql=sql, params=params, max_rows=max_rows
            )

        return AgentResult(
            agent=self.name,
            success=True,
            data={
                "sql": sql,
                "explanation": explanation,
                "params": params,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows) if rows is not None else None,
                "exec_summary": exec_summary,
            },
            message=(
                f"Generated SQL ({len(rows)} rows returned)"
                if rows is not None
                else "SQL generated (not executed)"
            ),
            metadata={"dialect": dialect, "schema_provided": bool(schema)},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute(
        self,
        db_path: str,
        sql: str,
        params: list[Any],
        max_rows: int,
    ) -> tuple[list[dict], list[str], str]:
        """Run the query against a local SQLite file."""
        if not Path(db_path).exists():
            self._log(f"DB file not found: {db_path}")
            return [], [], f"Database file not found: {db_path}"

        self._log(f"Executing query against {db_path}…")
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            raw_rows = cursor.fetchmany(max_rows)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            conn.close()
        except sqlite3.Error as exc:
            self._log(f"Execution error: {exc}")
            return [], [], f"SQL execution error: {exc}"

        rows = [dict(r) for r in raw_rows]
        self._log(f"Query returned {len(rows)} rows")

        if not rows:
            return rows, columns, "Query returned no rows."

        # Ask Claude to summarise the results
        preview = json.dumps(rows[:10], default=str, indent=2)
        summary_prompt = (
            f"Question: {sql}\n\nFirst {min(10, len(rows))} rows "
            f"(of {len(rows)} total):\n{preview}"
        )
        exec_summary = self._call_claude(_EXEC_SYSTEM, summary_prompt, max_tokens=512)
        return rows, columns, exec_summary
