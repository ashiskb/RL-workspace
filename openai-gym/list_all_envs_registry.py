#!/Users/ashis/venv-directory/venv-p310-RL-workspace/bin/python
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x list_all_envs_registry.py
#$ ./list_all_envs_registry.py

from pprint import pprint
from gymnasium import envs
#all_envs = envs.registry.all()
#env_ids = [env_spec.id for env_spec in all_envs]
#pprint(sorted(env_ids))
for key in envs.registry.keys():
    print(key)