import os
os.listdir('.')
targets = [f for f in os.listdir('.') if f.startswith('mc10c')]
f = open('../cmd.sh', 'r')
commands = f.readlines()
f.close()
commands = [cmd.strip() for cmd in commands if cmd.strip()]

targets.sort()
for t in targets:
    accuracy = [acc for acc in os.listdir(f'{t}') if acc.startswith('acc_')]
    for ln in range(len(commands)):
        if commands[ln].find(t)>=0:
            break
    print(f'{t}: {accuracy}, {commands[ln+1]}')
