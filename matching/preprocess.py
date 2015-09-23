import pandas as pd
from matching.profile import street_address


def business_name(name):
    return unicode(name.lower().translate(None,"'.,"), 'unicode-escape') if pd.notnull(name) else u''


def split_street_num(address_text, other=None):
    addr_type = street_address(address_text)
    if addr_type in ('STNUM', 'HSTNUM'):
        return address_text.split(' ', 1)[0]
    elif addr_type == 'POBOX':
        return address_text.split()[-1]
    else:
        return other if other else address_text
