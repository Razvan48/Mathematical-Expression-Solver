

def factor_evaluation(expression, index):
    result = 0

    if expression[index] == '(':
        index += 1
        result, index = expression_evaluation(expression, index)
        index += 1
    else:
        passed_the_decimal = False
        num_digits_after_decimal = 0
        while index < len(expression) and (expression[index].isdigit() or
                                           expression[index] == '.' or expression[index] == ','):
            if expression[index].isdigit():
                result = result * 10 + int(expression[index])

                if passed_the_decimal:
                    num_digits_after_decimal += 1
            else:  # expression[index] == '.' or expression[index] == ',':
                passed_the_decimal = True
            index += 1

        result /= (10 ** num_digits_after_decimal)

    return result, index


def term_evaluation(expression, index):
    result, index = factor_evaluation(expression, index)

    while index < len(expression) and (expression[index] == '*' or expression[index] == '/'):
        if expression[index] == '*':
            index += 1
            next_factor, index = factor_evaluation(expression, index)
            result *= next_factor
        else:  # expression[index] == '/':
            index += 1
            next_factor, index = factor_evaluation(expression, index)
            if next_factor == 0:
                raise ValueError('Error: Divided by 0')
            result /= next_factor

    return result, index


def expression_evaluation(expression, index):
    result, index = term_evaluation(expression, index)

    while index < len(expression) and (expression[index] == '+' or expression[index] == '-'):
        if expression[index] == '+':
            index += 1
            next_term, index = term_evaluation(expression, index)
            result += next_term
        else:  # expression[index] == '-':
            index += 1
            next_term, index = term_evaluation(expression, index)
            result -= next_term

    return result, index


def evaluate(expression, index=0):
    expression = expression.replace(' ', '')
    return expression_evaluation(expression, index)[0]


