from sparse_causal_model_learner_rl.configs.search_helpers.opt_cycle import get_stage


def test_get_stage():
    def check_one(lst, position, answer):
        computed = get_stage(position, lst)
        assert computed == answer, (lst, position, answer, computed)

    check_one([10, 10, 20], 0, 0)
    check_one([10, 10, 20], 1, 0)
    check_one([10, 10, 20], 9, 0)
    check_one([10, 10, 20], 10, 1)
    check_one([10, 10, 20], 19, 1)
    check_one([10, 10, 20], 20, 2)
    check_one([10, 10, 20], 29, 2)
    check_one([10, 10, 20], 30, 2)
    check_one([10, 10, 20], 39, 2)
    check_one([10, 10, 20], 40, 0)
