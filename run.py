import os
import subprocess
import sys


def main(file_path, output_path=None):
    # 获取文件后缀名
    file_ext = os.path.splitext(file_path)[1]

    # 根据文件后缀名选择相应的编译和运行命令
    if file_ext == '.java':
        cmd_compile = f'javac {file_path}'
        cmd_run = f'java {os.path.splitext(os.path.basename(file_path))[0]}'
    elif file_ext == '.cpp':
        cmd_compile = f'g++ {file_path} -o {os.path.splitext(file_path)[0]}'
        cmd_run = f'{os.path.splitext(file_path)[0]}'
    elif file_ext == '.py':
        cmd_compile = None
        cmd_run = f'python {file_path}'
    elif file_ext in ('.s', '.asm', '.MIPS'):
        cmd_compile = f'mips-linux-gnu-gcc -static {file_path} -o {os.path.splitext(file_path)[0]}'
        cmd_run = f'qemu-mips {os.path.splitext(file_path)[0]}'
    elif file_ext == '.sql':
        cmd_compile = None
        cmd_run = f'mysql -u root -p < {file_path}'
    else:
        print(f'Unsupported file extension: {file_ext}')
        return

    # 编译和运行
    if cmd_compile:
        subprocess.run(cmd_compile, shell=True, check=True)
    if output_path:
        with open(output_path, 'w') as f:
            subprocess.run(cmd_run, shell=True, check=True, stdout=f)
    else:
        subprocess.run(cmd_run, shell=True, check=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run.py <file> [<output_path>]')
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main(sys.argv[1], sys.argv[2])
