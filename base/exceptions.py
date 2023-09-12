class BotQuotaExceeded(Exception):
    def __init__(self, message="User has exceeded the assigned quota"):
        super().__init__(message)
