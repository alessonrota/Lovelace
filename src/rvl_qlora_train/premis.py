from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import append_jsonl, now_iso_utc


class PremisRecorder:
    def __init__(self, premis_dir: Path, run_id: str):
        self.premis_dir = premis_dir
        self.run_id = run_id
        self.events_file = premis_dir / "events.jsonl"
        self.objects_file = premis_dir / "objects.jsonl"
        self.agents_file = premis_dir / "agents.jsonl"
        self.rights_file = premis_dir / "rights.jsonl"
        self.premis_dir.mkdir(parents=True, exist_ok=True)
        self.event_counter = self._infer_event_counter()

    def _infer_event_counter(self) -> int:
        if not self.events_file.exists():
            return 0
        count = 0
        with self.events_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def ensure_defaults(self, base_model: str) -> None:
        if not self.agents_file.exists() or self.agents_file.stat().st_size == 0:
            self.add_agent("agent:pipeline", "rvl_qlora_train_pipeline", "software")
            self.add_agent("agent:paddleocr", "paddleocr", "software")
            self.add_agent("agent:trainer", "transformers+peft", "software")
            self.add_agent("agent:base_model", base_model, "software")

        if not self.rights_file.exists() or self.rights_file.stat().st_size == 0:
            append_jsonl(
                self.rights_file,
                {
                    "rightsStatementIdentifier": {
                        "rightsStatementIdentifierType": "local",
                        "rightsStatementIdentifierValue": f"rights:{self.run_id}",
                    },
                    "rightsBasis": "copyright",
                    "rightsGranted": [
                        {
                            "act": "analyze",
                            "restriction": "evaluation-and-training-only",
                            "termOfGrant": "run-scope",
                        }
                    ],
                    "createdAt": now_iso_utc(),
                },
            )

    def add_agent(self, agent_id: str, name: str, agent_type: str) -> None:
        append_jsonl(
            self.agents_file,
            {
                "agentIdentifier": {
                    "agentIdentifierType": "local",
                    "agentIdentifierValue": agent_id,
                },
                "agentName": name,
                "agentType": agent_type,
                "createdAt": now_iso_utc(),
            },
        )

    def add_object(self, object_id: str, path: str, category: str = "file", fmt: str = "unknown") -> None:
        append_jsonl(
            self.objects_file,
            {
                "objectIdentifier": {
                    "objectIdentifierType": "local",
                    "objectIdentifierValue": object_id,
                },
                "objectCategory": category,
                "objectPath": path,
                "objectCharacteristics": {"format": fmt},
                "createdAt": now_iso_utc(),
            },
        )

    def add_event(
        self,
        event_type: str,
        outcome: str,
        detail: dict[str, Any],
        object_id: str | None = None,
        object_path: str | None = None,
        agents: list[str] | None = None,
    ) -> None:
        self.event_counter += 1
        event_id = f"evt-{self.event_counter:06d}"

        payload: dict[str, Any] = {
            "eventIdentifier": {
                "eventIdentifierType": "local",
                "eventIdentifierValue": event_id,
            },
            "eventType": event_type,
            "eventDateTime": now_iso_utc(),
            "eventOutcomeInformation": {
                "eventOutcome": outcome,
                "eventOutcomeDetail": detail,
            },
        }

        if object_id or object_path:
            payload["linkingObjectIdentifier"] = [
                {
                    "linkingObjectIdentifierType": "local",
                    "linkingObjectIdentifierValue": object_id or object_path or "",
                }
            ]

        if agents:
            payload["linkingAgentIdentifier"] = [
                {
                    "linkingAgentIdentifierType": "local",
                    "linkingAgentIdentifierValue": agent,
                }
                for agent in agents
            ]

        append_jsonl(self.events_file, payload)
