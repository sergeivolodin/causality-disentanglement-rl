import gin
import logging

def get_stage(position, lst):
    """Return the first index such that cumsum(<=i) >= position."""
    i = 0
    summed = 0
    total = sum(lst)
    position = position % total
    for value in lst:
        if summed + value > position:
            return i
        else:
            summed = summed + value
            i = i + 1
    return len(lst) - 1


@gin.configurable
def opt_active_cycle(learner, opt_key, opts_list=None, burst_sizes=None, **kwargs):
    """Given an epoch, determine if an optimizer should be active."""

    epoch = learner.epochs
    if opt_key not in opts_list: # unknown optimizers are always active
        return True

    stage = get_stage(epoch, [burst_sizes[k] for k in opts_list])

    opt_index = opts_list.index(opt_key)
    active = opt_index == stage

#    if active:
    logging.warning(f"Epoch {epoch} Stage {stage} optimizer {opt_key} active={active}")

    return active
