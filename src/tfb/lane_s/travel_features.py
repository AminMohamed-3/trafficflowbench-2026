"""F2 canonical information plus resolved travel-time characteristics."""
from .gap_features import day_features as parent_features
from .characteristics import day_characteristics

def day_features(cache,day,bases,panel_code,gap_context=None):
    fields,override=parent_features(cache,day,bases,panel_code,gap_context)
    fields.update(day_characteristics(cache,day))
    assert len(fields)==177
    return fields,override

