from .kv_teams import transform as kv_teams
from .rank_routing import transform as rank_routing


def install(module, profile, post_transform=None):
    original = module._metal
    captures = {}

    def build(name, **options):
        if name == "flint_decode":
            assert f"if({profile.mlp_workers}=={profile.mlp_workers})" in options["source"], (
                "Own-rank routing requires the expert-team schedule"
            )
            header, body = rank_routing(options["header"], options["source"], profile.mlp_workers)
            header, body = kv_teams(header, body, profile.workers)
            if post_transform is not None:
                header, body = post_transform(header, body)
            options = dict(options, header=header, source=body)
            captures["decode"] = dict(options)
            name += "_resident_teams"
        return original(name, **options)

    module._metal = build
    return captures
