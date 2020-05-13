import os
import gc
import subprocess
import logging


logger = logging.getLogger(__name__)


def execute_command(command_list, show_stdout=False, show_stderr=False, show_command=False, no_fork=False, stdin=None,
                    file_input=None):
    try:
        if no_fork:
            os.system(" ".join(command_list))
        else:
            stdout, stderr = subprocess_execute(command_list, stdin, file_input)
            if show_stdout:
                logger.info("stdout:\n{stdout}".format(stdout=stdout))
            if show_stderr:
                logger.info("stderr:\n{stderr}".format(stderr=stderr))
    except:
        raise
    finally:
        if show_command:
            logger.info("command:\n{command}".format(command=" ".join(command_list)))


def subprocess_execute(command_list, std_input=None, file_input=None):
    gc.collect()
    popen_obj = subprocess.Popen(command_list, stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if file_input: popen_obj.stdin.write('@{0}'.format(file_input))
    if std_input: popen_obj.stdin.write(std_input)
    stdout, stderr = popen_obj.communicate()
    return_code = popen_obj.wait()
    if return_code != 0:
        error_message = get_error_message(command_list, stdout, stderr)
        logger.error(error_message)
        raise Exception(error_message)
    return stdout, stderr


def get_error_message(command_list, stdout="", stderr=""):
    error_message = \
'''
command execution failed:
    command:
{command}
'''.format(command=" ".join(command_list))
    if stdout:
        error_message += \
'''
    stdout:
{stdout}
'''.format(stdout=stdout)
    if stderr:
        error_message += \
'''
    stderr:
{stderr}
'''.format(stderr=stderr)
    return error_message
