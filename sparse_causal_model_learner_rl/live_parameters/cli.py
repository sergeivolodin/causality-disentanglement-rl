import ray
import argparse
import inquirer


parser = argparse.ArgumentParser(description="Change parameters for running trials")
parser.add_argument('--trial_communicator', type=str, required=True,
                    help="Ray actor name with the parameter communicator")

if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(address='auto')   # connect to the server that has been started by main.py

    communicator = ray.get_actor(args.trial_communicator)

    while True:
        questions = [
            inquirer.List('command',
                          message="What do you want to do?",
                          choices=['Edit parameters', 'Exit'],
                          ),
        ]
        answers = inquirer.prompt(questions)
        if not answers:
            continue

        if answers['command'] == 'Exit':
            break
        elif answers['command'] == 'Edit parameters':
            current_parameters = ray.get(communicator.get_current_parameters.remote())

            options = [(f"{param_name} {param_value}", param_name) for param_name, param_value
                       in current_parameters.items()]

            questions = [
                inquirer.List('param_name',
                              message="Which parameter to edit?",
                              choices=options,
                              ),
            ]
            answers = inquirer.prompt(questions)
            if not answers:
                continue
            param_name = answers['param_name']

            old_value = current_parameters[param_name]


            questions = [
                inquirer.Text(param_name,
                              message=f"New value for {param_name} [now {old_value}]"),
            ]

            answers = inquirer.prompt(questions)
            if not answers:
                continue

            new_value = answers[param_name]
            new_value = type(old_value)(new_value)

            questions = [
                inquirer.Confirm('continue',
                                 message=f"Replace {old_value} with {new_value} for {param_name}?"),
            ]

            answers = inquirer.prompt(questions)
            if not answers:
                continue
            if answers['continue']:
                ray.get(communicator.update_parameter.remote(param_name, new_value))
                print(f"Parameter {param_name} set to {new_value} (was {old_value})")
            else:
                print("Change cancelled.")