def __(__):
    if e['r'] + e['f'] > 1e-2:
        if hypers['s'] > 1e-5:
            suggested_hyper = hypers['s'] * 0.5
    else:
        if hypers['s'] < 10:
            suggested_hyper = hypers['s'] / 0.5

    if i - last_hyper_adjustment >= 100:
        hypers['s'] = suggested_hyper
        last_hyper_adjustment = i