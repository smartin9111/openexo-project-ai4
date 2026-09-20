from src.environment.openexo_env import OpenExoEnvironment


env = OpenExoEnvironment()

print("OpenExo loaded successfully!")
print(env.get_model_info())