from gin_tune import tune_gin
from myfunc import f
import gin


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    analysis = tune_gin(f)

    print("Sum results")
    print(sorted([y['sum'] for y in analysis.results.values()]))
