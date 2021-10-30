import gin


@gin.configurable
def part(item, key):
    return item[key]
