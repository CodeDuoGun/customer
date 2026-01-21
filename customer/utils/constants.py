"""
Constants and speeches for the chat service.
"""
from enum import Enum


class MessageEventStatus(Enum):
    """Message event status enumeration."""
    DELTA = "delta"
    COMPLETED = "completed"


class Speeches:
    """Predefined speech responses."""

    # No doctor found speech
    NoDoctorSpeech = """很抱歉，小惠暂时没有找到符合您条件的医生。您可以描述一下您的具体症状，小惠会尽力为您推荐合适的医生哦～"""

    # Doctor recommendation speech
    RecommendDoctorSpeech = """根据您的描述，小惠为您推荐以下医生："""

    # AI warning speech
    AIWarningSpeech = """以上内容由AI生成，仅供参考。如有就医需求，请以医生建议为准。小惠会持续学习，为您提供更准确的服务！"""

    # Doctor introduction template
    DoctorIntroTemplate = """
医生：{name}
{introduction}
擅长：{specialty}
{location}

如果您需要了解更多信息或预约挂号，可以联系小惠哦～还有其他想了解的吗？
""".strip()


class IntentTypes:
    """Intent type constants."""
    DISEASE_SEARCH = "disease_search"
    DOCTOR_SEARCH = "doctor"
    QA_SEARCH = "qa"
    DOCTOR_INTRODUCE = "introduce"
    OTHER = "other"
