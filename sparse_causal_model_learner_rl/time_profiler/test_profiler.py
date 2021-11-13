from sparse_causal_model_learner_rl.time_profiler.profiler import TimeProfiler
from time import sleep


def test_profiler_output():
    p = TimeProfiler(enable=True)


    p.start('test1')
    p.start('test11')
    sleep(0.01)
    p.end('test11')
    p.start('test12')
    p.start('test123')
    sleep(0.01)
    p.end('test123')
    p.end('test12')
    p.end('test1')
    p.start('test2')
    sleep(0.01)
    p.end('test2')


    def traverse_to_json(item):
        result = {}
        # result['delta'] = item.delta()
        result['name'] = item.name
        result['children'] = []
        for sub_item in item.children:
            result['children'].append(traverse_to_json(sub_item))
        return result

    def only_name(data):
        return {data['name']: [only_name(t) for t in data['children']]}

    p.report()
    data = p.nested()
    data = only_name(traverse_to_json(data))
    data_expected = {'profiler': [{'profiler_test1': [{'profiler_test1_test11': []}, {'profiler_test1_test12': [{'profiler_test1_test12_test123': []}]}]}, {'profiler_test2': []}]}
    assert data == data_expected
