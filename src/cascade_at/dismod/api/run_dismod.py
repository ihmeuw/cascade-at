import subprocess
import sys
from types import SimpleNamespace
from typing import List
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def run_dismod(dm_file: str, command: str):
    """
    Executes a command on a dismod file.

    Parameters
    ----------
    dm_file
        the dismod db filepath
    command
        a command to run
    """
    def check_last_command(dm_file: str, command: str):
        LOG.warning("FIXME -- GMA -- I need an entirely new way to wrap dmdismod with a preprocessor to handle the ODE strategy.")
        LOG.warning("FIXME -- GMA -- Check_last_command needs to wrap the call to dmdismod, not the ODE preprocessor.")
        from cascade_at.dismod.api.dismod_io import DismodIO
        db = DismodIO(dm_file)
        log = db.log
        last_begin = [l for i,l in log.iterrows()
                      if l.message_type == 'command'
                      and l.message.startswith('begin ')]
        rtn = True
        if not last_begin:
            LOG.error(f"ERROR: Failed to find a 'begin' command.")
            rtn = False
        else:
            last_begin = last_begin[-1]
        if rtn:
            start_cmd = [l for i,l in log[last_begin.log_id:].iterrows()
                         if l.message_type == 'command'
                         and l.message.startswith(f'begin {command}')]
            if not start_cmd:
                LOG.error(f"ERROR: Expected 'begin {command}' but found '{last_begin.message}'.")
                rtn = False
            else:
                start_cmd = start_cmd[-1]
        if rtn:
            end_cmd = [l for i,l in log[start_cmd.log_id:].iterrows()
                       if l.message_type == 'command'
                       and l.message.startswith(f'end {command}')]
            if not end_cmd:
                LOG.error(f"ERROR: Did not find end for this '{start_cmd.message}' command")
                rtn = False
            for i,l in log[start_cmd.log_id:].iterrows():
                if l.message_type in ['error', 'warning']:
                    LOG.info (f"DISMOD {l.message_type}: {l.message.rstrip()}")
                    rtn = False
        if rtn:
            LOG.info (f"{command} OK")
        else:
            LOG.error (f"ERROR: {command} had errors, warnings, or failed to complete.")
        return rtn

    command = ["dmdismod", str(dm_file), command]
    command = ' '.join(command)
    LOG.info(f"Running {command}")

    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    info = SimpleNamespace()
    info.exit_status = process.returncode
    info.stdout = process.stdout.decode()
    info.stderr = process.stderr.decode()
    
    # Remove the ODE overloading
    dismod_command = command.replace(' ODE ', ' ').split()[0]
    check_dismod = check_last_command(dm_file, dismod_command)

    return info


def run_dismod_commands(dm_file: str, commands: List[str], sys_exit=True):
    """
    Runs multiple commands on a dismod file and returns the exit statuses.
    Will raise an exception if it runs into an error.

    Parameters
    ----------
    dm_file
        the dismod db filepath
    commands
        a list of strings
    sys_exit
        whether to exit the code altogether if there is an error. If False,
        then it will pass the error string back to the original python process.
    """
    def ipopt_output_logger(msg):
        # See https://coin-or.github.io/Ipopt/OUTPUT.html
        LOG.info('Checking Ipopt output for errors')
        if (('optimal solution found' in msg) or
            ('solved to acceptable level' in msg)):
            return LOG.info
        if ('maximum number of iterations exceeded') in msg:
            return LOG.warning
        # Handle other possible Ipopt console messages?
        msg = msg.replace('overall nlp error', 'overall nlp      ')
        if msg.count('error') > 0:
            return LOG.error
        return LOG.info

    def log_stream(stream, exit_status):
        "Log the dismod return messages appropriately"
        msg = stream.lower()
        if exit_status:
            log = LOG.error
        else:
            if 'ipopt' in msg:
                log = ipopt_output_logger(msg)
            elif 'error' in msg:
                log = LOG.error
            elif 'warn' in msg:
                log = LOG.warning
            else:
                log = LOG.info
        for i,line in enumerate(stream.splitlines()):
            if line:
                log(line)

    processes = dict()
    if isinstance(commands, str):
        commands = [commands]
    for c in commands:
        process = run_dismod(dm_file=dm_file, command=c)
        processes.update({c: process})
        if process.exit_status:
            LOG.error(f"{c} failed with exit_status {process.exit_status}:")
            log_stream(process.stderr, process.exit_status)
            log_stream(process.stdout, process.exit_status)
            if sys_exit:
                try:
                    raise RuntimeError(
                        f"Dismod-AT failed with exit status {process.exit_status}."
                        f"Exiting program."
                    )
                except RuntimeError:
                    sys.exit(process.exit_status)
        else:
            log_stream(process.stderr, process.exit_status)
            log_stream(process.stdout, process.exit_status)
    return processes
