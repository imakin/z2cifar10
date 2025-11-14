import os
import subprocess
# makin 2025

def summarize(target):
    report = f"{target}/{target}/sol1/syn/report/csynth.rpt"
    if os.path.exists(report):
        with open(report, 'r') as f:
            lines = f.readlines()
        # barisan = []
        cycles = 0
        for i in range(len(lines)):
            # ada 3 komponen di level "    |  + ", jumlahkan cycle mereka
            if lines[i].startswith(f'    |  + '):
                # print(lines[i])
                # barisan.append(lines[i])
                cycle = int(lines[i].split('|')[4].strip())
                cycles += cycle
        acc_filename = [f for f in os.listdir(f'{target}') if f.startswith('acc_')]
        
        # check fmax 
        logs = f"{target}/logs/hls_run_tcl.log"
        fmax = "N/A"
        try:
            with open(logs, 'r') as f:
                loglines = f.readlines()
            # cari baris yang mengandung "Estimated Fmax"
            for line in loglines:
                if "Estimated Fmax:" in line:
                    fmax = line.split(':')[2].strip()
                    break
        except FileNotFoundError:
            pass


        # check sparsity
        cmd = f'cd {target}/firmware/weights && for f in w*.txt; do total=$(wc -w < "$f"); zeros=$(grep -o "0\.000000000" "$f" | wc -l); if [ $total -gt 0 ]; then pct=$(echo "scale=2; $zeros * 100 / $total" | bc); echo "$f: $zeros/$total zeros ($pct%)"; fi; done'
        cmd = f'cd {target}/firmware/weights && for f in w*.txt; do total=$(wc -w < "$f"); zeros=$(grep -o "0\.000000000" "$f" | wc -l); if [ $total -gt 0 ]; then echo "$f:$zeros/$total"; fi; done'
        sparsity_info = subprocess.getoutput(cmd).replace('\n', ', ')
        if acc_filename:
            print(f"{target}: {acc_filename} : {cycles} cycles. Est Fmax: {fmax}.\n\tsparsity:{sparsity_info}")

    else:pass



targets = [f for f in os.listdir('.') if f.startswith('mc10c')]
targets.sort()
for t in targets:
    summarize(t)