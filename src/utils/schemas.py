from pydantic import BaseModel, Field


status_only_schema = """{
            "type": "object",
            "properties": {
                "Status": {"type": "integer",
                           "enum": [0,1]
                           }
            },
            "required": ["Status"]
}"""

status_n_reason_schema = """{
            "type": "object",
            "properties": {
                "Status": {"type": "integer",
                           "enum": [0,1]
                           },
                "Clinical Reason": {"type": "string"},
            },
            "required": ["Status", "Clinical Reason",]
}"""


class status_only_schema(BaseModel):
    Status: int = Field(description="Presence or absence of a condition, 1: Present, 0: Absent")


class status_n_reason_schema(BaseModel):
    Status: int = Field(description="Presence or absence of a condition, 1: Present, 0: Absent")
    Clinical_Reason: str = Field(description="Clinical reason for Status value")