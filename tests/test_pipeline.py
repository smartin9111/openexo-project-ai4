from src.perception.dummy import DummyPerception
from src.controllers.dummy import DummyController


observation = {
    "joint_positions": [0.1, 0.2, 0.3],
    "joint_velocities": [0.01, 0.02, 0.03],
    "muscle_activations": [0.4, 0.5],
    "foot_contacts": [1, 0],
}


perception = DummyPerception()
controller = DummyController()

state = perception.process(observation)
action = controller.compute_action(state)


print("Raw observation:")
print(observation)

print("\nPerception state:")
print(state)

print("\nController action:")
print(action)