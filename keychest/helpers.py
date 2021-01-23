def fill_n(arr, offset_x, value, symbol=True):
    """Fill cells of arr starting from row offset_x with value in unary counting."""
    value_left = value
    width = arr.shape[1]
    current_row = offset_x
    while value_left:
        add_this_iter = min(width, value_left)
        arr[current_row, :add_this_iter] = [symbol] * add_this_iter
        current_row += 1
        value_left -= add_this_iter