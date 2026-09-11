"""Application errors with stable structured HTTP projections."""


class ProviderUnavailable(ValueError):
    def __init__(self, provider_ids: tuple[str, ...]):
        self.provider_ids = provider_ids
        super().__init__("provider credentials are unavailable")
