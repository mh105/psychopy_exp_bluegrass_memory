#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.2a1),
    on Wed Oct 16 13:55:38 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, iohub
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from eeg
import pyxid2
import threading
import signal


def exit_after(s):
    '''
    function decorator to raise KeyboardInterrupt exception
    if function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, signal.raise_signal, args=[signal.SIGINT])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@exit_after(1)  # exit if function takes longer than 1 seconds
def _get_xid_devices():
    return pyxid2.get_xid_devices()


def get_xid_devices():
    print("Getting a list of all attached XID devices...")
    attempt_count = 0
    while attempt_count >= 0:
        attempt_count += 1
        print('     Attempt:', attempt_count)
        attempt_count *= -1  # try to exit the while loop
        try:
            devices = _get_xid_devices()
        except KeyboardInterrupt:
            attempt_count *= -1  # get back in the while loop
    return devices


devices = get_xid_devices()

if devices:
    dev = devices[0]
    print("Found device:", dev)
    assert dev.device_name == 'Cedrus C-POD', "Incorrect C-POD detected."
    dev.set_pulse_duration(50)  # set pulse duration to 50ms

    # Start EEG recording
    print("Sending trigger code 126 to start EEG recording...")
    dev.activate_line(bitmask=126)  # trigger 126 will start EEG
    print("Waiting 10 seconds for the EEG recording to start...")
    print("")
    core.wait(10)  # wait 10s for the EEG system to start recording

    # Marching lights test
    print("C-POD<->eego 7-bit trigger lines test...")
    for line in range(1, 8):  # raise lines 1-7 one at a time
        print("  raising line {} (bitmask {})".format(line, 2 ** (line-1)))
        dev.activate_line(lines=line)
        core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.con.set_digio_lines_to_mask(0)  # XidDevice.clear_all_lines()
    print("EEG system is now ready for the experiment to start.")

else:
    # Dummy XidDevice for code components to run without C-POD connected
    class dummyXidDevice(object):
        def __init__(self):
            pass
        def activate_line(self, lines=None, bitmask=None):
            pass


    print("WARNING: No C-POD connected for this session! "
          "You must start/stop EEG recording manually!")
    dev = dummyXidDevice()

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.2a1'
expName = 'bluegrass_memory'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['session'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/alexhe/Dropbox (Personal)/Active_projects/PsychoPy/exp_bluegrass_memory/bluegrass_memory.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # log the filename of last_app_load.log
    print('target_last_app_load_log_file: ' + filename + '_last_app_load.log')
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('debug')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup eyetracking
    ioConfig['eyetracker.eyelink.EyeTracker'] = {
        'name': 'tracker',
        'model_name': 'EYELINK 1000 DESKTOP',
        'simulation_mode': False,
        'network_settings': '100.1.1.1',
        'default_native_data_file_name': 'EXPFILE',
        'runtime_settings': {
            'sampling_rate': 1000.0,
            'track_eyes': 'LEFT_EYE',
            'sample_filtering': {
                'FILTER_FILE': 'FILTER_LEVEL_OFF',
                'FILTER_ONLINE': 'FILTER_LEVEL_OFF',
            },
            'vog_settings': {
                'pupil_measure_types': 'PUPIL_DIAMETER',
                'tracking_mode': 'PUPIL_CR_TRACKING',
                'pupil_center_algorithm': 'ELLIPSE_FIT',
            }
        }
    }
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    deviceManager.devices['eyetracker'] = ioServer.getDevice('tracker')
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_welcome') is None:
        # initialise key_welcome
        key_welcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_welcome',
        )
    # create speaker 'read_welcome'
    deviceManager.addDevice(
        deviceName='read_welcome',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_et') is None:
        # initialise key_et
        key_et = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_et',
        )
    # create speaker 'read_et'
    deviceManager.addDevice(
        deviceName='read_et',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'read_start'
    deviceManager.addDevice(
        deviceName='read_start',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct') is None:
        # initialise key_instruct
        key_instruct = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct',
        )
    # create speaker 'read_instruct'
    deviceManager.addDevice(
        deviceName='read_instruct',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_schematic') is None:
        # initialise key_schematic
        key_schematic = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_schematic',
        )
    # create speaker 'read_schematic'
    deviceManager.addDevice(
        deviceName='read_schematic',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_practice') is None:
        # initialise key_practice
        key_practice = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_practice',
        )
    # create speaker 'read_practice'
    deviceManager.addDevice(
        deviceName='read_practice',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_practice_repeat') is None:
        # initialise key_practice_repeat
        key_practice_repeat = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_practice_repeat',
        )
    # create speaker 'read_practice_repeat'
    deviceManager.addDevice(
        deviceName='read_practice_repeat',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_catch_delay') is None:
        # initialise key_resp_catch_delay
        key_resp_catch_delay = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_catch_delay',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_checkpoint') is None:
        # initialise key_checkpoint
        key_checkpoint = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_checkpoint',
        )
    # create speaker 'read_checkpoint'
    deviceManager.addDevice(
        deviceName='read_checkpoint',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_begin') is None:
        # initialise key_begin
        key_begin = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_begin',
        )
    # create speaker 'read_begin'
    deviceManager.addDevice(
        deviceName='read_begin',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_rest') is None:
        # initialise key_rest
        key_rest = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_rest',
        )
    # create speaker 'read_rest'
    deviceManager.addDevice(
        deviceName='read_rest',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_continue') is None:
        # initialise key_continue
        key_continue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_continue',
        )
    # create speaker 'read_continue'
    deviceManager.addDevice(
        deviceName='read_continue',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'read_thank_you'
    deviceManager.addDevice(
        deviceName='read_thank_you',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "_welcome" ---
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Welcome! This task will take approximately 30 minutes.\n\nBefore we explain the task, we need to first calibrate the eyetracking camera. Please sit in a comfortable position with your head on the chin rest. Once we begin, it is important that you stay in the same position throughout this task.\n\nPlease take a moment to adjust the chair height, chin rest, and sitting posture. Make sure that you feel comfortable and can stay still for a while.\n\n\nWhen you are ready, press any of the white keys to begin',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome = keyboard.Keyboard(deviceName='key_welcome')
    read_welcome = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_welcome',    name='read_welcome'
    )
    read_welcome.setVolume(1.0)
    
    # --- Initialize components for Routine "_et_instruct" ---
    text_et = visual.TextStim(win=win, name='text_et',
        text='During the calibration, you will see a target circle moving around the screen. Please try to track it with your eyes.\n\nMake sure to keep looking at the circle when it stops, and follow it when it moves. It is important that you keep your head on the chin rest once this part begins.\n\n\nPress any of the white keys when you are ready, and our team will start the calibration for you',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_et = keyboard.Keyboard(deviceName='key_et')
    read_et = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_et',    name='read_et'
    )
    read_et.setVolume(1.0)
    
    # --- Initialize components for Routine "_et_mask" ---
    text_mask = visual.TextStim(win=win, name='text_mask',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "__start__" ---
    text_start = visual.TextStim(win=win, name='text_start',
        text='We are now ready to begin...',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    read_start = sound.Sound(
        'A', 
        secs=1.8, 
        stereo=True, 
        hamming=True, 
        speaker='read_start',    name='read_start'
    )
    read_start.setVolume(1.0)
    # Run 'Begin Experiment' code from trigger_table
    ##TASK ID TRIGGER VALUES##
    # special code 100 (task start, task ID should follow immediately)
    task_start_code = 100
    # special code 103 (task ID for bluegrass memory task)
    task_ID_code = 103
    
    ##GENERAL TRIGGER VALUES##
    # special code 122 (block start)
    block_start_code = 122
    # special code 123 (block end)
    block_end_code = 123
    
    ##TASK SPECIFIC TRIGGER VALUES##
    # N.B.: only use values 1-99 and provide clear comments on used values
    target_start_code = 2
    test_start_code = 3
    
    # Run 'Begin Experiment' code from task_id
    dev.activate_line(bitmask=task_start_code)  # special code for task start
    core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.activate_line(bitmask=task_ID_code)  # special code for task ID
    
    etRecord = hardware.eyetracker.EyetrackerControl(
        tracker=eyetracker,
        actionType='Start Only'
    )
    # Run 'Begin Experiment' code from condition_setup
    # Set up condition arrays for the experiment
    rng = np.random.default_rng()
    session_int = int(expInfo['session'])
    idx_start = (session_int - 1) * 124 + 1
    idx_end = session_int * 124 + 1
    image_filenames = rng.permutation(['resource/' + str(x).zfill(3) + '.bmp' for x in range(idx_start, idx_end)])
    n_objects_per_trial = 2  # each trial alternates between two object images
    n_images_per_trial = 5  # each trial will present 5 test images of the two objects
    n_match_options = [2, 3]  # target image can appear 2 or 3 times per trial
    
    # Practice trials
    n_trials_practice = 2
    n_objects_practice = n_trials_practice * n_objects_per_trial  # 4 objects during practice
    image_fn_practice = image_filenames[:n_objects_practice]
    image_fn_practice_list = [image_fn_practice[i * n_objects_per_trial:(i + 1) * n_objects_per_trial] for i in range(n_trials_practice)]
    # decide the type of each image in the sequence of each trial
    imageType_practice_list = []
    for _ in range(n_trials_practice):
        n_match = rng.choice(n_match_options)
        n_nonmatch = n_images_per_trial - n_match
        imageType_practice_list.append(rng.permutation([1] * n_match + [0] * n_nonmatch))  # 1 = match, 0 = non-match
    
    # Main trials
    n_blocks = 2
    n_trials_per_block = 30  # a total of 60 trials
    n_trials = n_blocks * n_trials_per_block
    n_objects = n_trials * n_objects_per_trial  # 120 objects during main experiment
    image_fn = image_filenames[n_objects_practice:n_objects_practice + n_objects]
    image_fn_list = [image_fn[i * n_objects_per_trial:(i + 1) * n_objects_per_trial] for i in range(n_trials)]
    # decide the type of each image in the sequence of each trial
    imageType_list = []
    for _ in range(n_trials):
        n_match = rng.choice(n_match_options)
        n_nonmatch = n_images_per_trial - n_match
        imageType_list.append(rng.permutation([1] * n_match + [0] * n_nonmatch))  # 1 = match, 0 = non-match
    # split the lists into blocks
    image_fn_list_blocks = [image_fn_list[i * n_trials_per_block:(i + 1) * n_trials_per_block] for i in range(n_blocks)]
    imageType_list_blocks = [imageType_list[i * n_trials_per_block:(i + 1) * n_trials_per_block] for i in range(n_blocks)]
    
    assert len(image_filenames) == 124, "Incorrect number of picture stimuli loaded for this session."
    assert len(image_filenames) / n_objects_per_trial == (n_trials_practice + n_trials), "Incorrect number of trials for this session."
    
    
    # --- Initialize components for Routine "instruct" ---
    text_instruct = visual.TextStim(win=win, name='text_instruct',
        text='In this task, you will first see a target image shown in a green box. Then, you will see a series of images presented one at a time following the target image.\n\nFor each image in the sequence, you need to decide whether it matches the target or not. If it matches the target, please press the Green key, and if it does not match the target, press the Red key.\n\nThere will be a different target image for each sequence, and the target image at the beginning of a new sequence is marked by the green box.\n\n\nPress any of the white keys to see an overview diagram',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct = keyboard.Keyboard(deviceName='key_instruct')
    read_instruct = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct',    name='read_instruct'
    )
    read_instruct.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_diagram" ---
    image_schematic = visual.ImageStim(
        win=win,
        name='image_schematic', 
        image='resource/taskSchematic.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    text_schematic = visual.TextStim(win=win, name='text_schematic',
        text='Press any of the white keys to continue',
        font='Arial',
        units='norm', pos=(0, -0.75), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_schematic = keyboard.Keyboard(deviceName='key_schematic')
    read_schematic = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_schematic',    name='read_schematic'
    )
    read_schematic.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_practice" ---
    text_practice = visual.TextStim(win=win, name='text_practice',
        text='We will first do a round of practice trials.\n\nRemember: if an image matches the target, press the Green key, and if it does not match the target, press the Red key.\n\nPlease respond as quickly and accurately as possible after you see each image.\n\n\nPress the green key to begin',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_practice = keyboard.Keyboard(deviceName='key_practice')
    read_practice = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_practice',    name='read_practice'
    )
    read_practice.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_practice_repeat" ---
    test_practice_repeat = visual.TextStim(win=win, name='test_practice_repeat',
        text='We will repeat the practice trials one more time.\n\nRemember: if an image matches the target, press the Green key, and if it does not match the target, press the Red key.\n\nPlease respond as quickly and accurately as possible after you see each image.\n\n\nPress the green key to begin',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_practice_repeat = keyboard.Keyboard(deviceName='key_practice_repeat')
    read_practice_repeat = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_practice_repeat',    name='read_practice_repeat'
    )
    read_practice_repeat.setVolume(1.0)
    
    # --- Initialize components for Routine "practice_setup" ---
    
    # --- Initialize components for Routine "trial_target" ---
    green_square_part1 = visual.Rect(
        win=win, name='green_square_part1',
        width=(0.42, 0.42)[0], height=(0.42, 0.42)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, 0.4824, -0.8353], fillColor=[-1.0000, 0.4824, -0.8353],
        opacity=None, depth=0.0, interpolate=True)
    green_square_part2 = visual.Rect(
        win=win, name='green_square_part2',
        width=(0.4, 0.4)[0], height=(0.4, 0.4)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    image_target = visual.ImageStim(
        win=win,
        name='image_target', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-2.0)
    # Run 'Begin Experiment' code from adjust_image_target_size
    def scale_to_size(image_object, max_size):
        image_object.size *= max_size / max(image_object.size)  # scale to max size
        image_object._requestedSize = None  # reset for next image original size
    
    key_resp_catch_delay = keyboard.Keyboard(deviceName='key_resp_catch_delay')
    text_fixation_first = visual.TextStim(win=win, name='text_fixation_first',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "trial_image" ---
    white_square = visual.Rect(
        win=win, name='white_square',
        width=(0.4, 0.4)[0], height=(0.4, 0.4)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=None, fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    image_test = visual.ImageStim(
        win=win,
        name='image_test', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-2.0)
    text_fixation = visual.TextStim(win=win, name='text_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "response_feedback" ---
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "practice_checkpoint" ---
    text_checkpoint = visual.TextStim(win=win, name='text_checkpoint',
        text='Please give us a moment to check whether practice trials need to be repeated...',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_checkpoint = keyboard.Keyboard(deviceName='key_checkpoint')
    read_checkpoint = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_checkpoint',    name='read_checkpoint'
    )
    read_checkpoint.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_begin" ---
    text_begin = visual.TextStim(win=win, name='text_begin',
        text='Great job! We will now start the task.\n\nRemember: if an image matches the target, press the Green key, and if it does not match the target, press the Red key.\n\nPlease respond as quickly and accurately as possible after you see each image. You will no longer receive feedback on your responses.\n\n\nPress the green key to begin',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_begin = keyboard.Keyboard(deviceName='key_begin')
    read_begin = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_begin',    name='read_begin'
    )
    read_begin.setVolume(1.0)
    
    # --- Initialize components for Routine "block_setup" ---
    
    # --- Initialize components for Routine "trial_setup" ---
    
    # --- Initialize components for Routine "trial_target" ---
    green_square_part1 = visual.Rect(
        win=win, name='green_square_part1',
        width=(0.42, 0.42)[0], height=(0.42, 0.42)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, 0.4824, -0.8353], fillColor=[-1.0000, 0.4824, -0.8353],
        opacity=None, depth=0.0, interpolate=True)
    green_square_part2 = visual.Rect(
        win=win, name='green_square_part2',
        width=(0.4, 0.4)[0], height=(0.4, 0.4)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    image_target = visual.ImageStim(
        win=win,
        name='image_target', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-2.0)
    # Run 'Begin Experiment' code from adjust_image_target_size
    def scale_to_size(image_object, max_size):
        image_object.size *= max_size / max(image_object.size)  # scale to max size
        image_object._requestedSize = None  # reset for next image original size
    
    key_resp_catch_delay = keyboard.Keyboard(deviceName='key_resp_catch_delay')
    text_fixation_first = visual.TextStim(win=win, name='text_fixation_first',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "trial_image" ---
    white_square = visual.Rect(
        win=win, name='white_square',
        width=(0.4, 0.4)[0], height=(0.4, 0.4)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=None, fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    image_test = visual.ImageStim(
        win=win,
        name='image_test', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-2.0)
    text_fixation = visual.TextStim(win=win, name='text_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "block_cleanup" ---
    
    # --- Initialize components for Routine "instruct_rest" ---
    text_rest = visual.TextStim(win=win, name='text_rest',
        text='Now you can take a break!\n\nAfter the short break, we will repeat the eyetracking camera calibration and continue with the rest of the task.\n\n\nPress any of the white keys when you are ready',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_rest = keyboard.Keyboard(deviceName='key_rest')
    read_rest = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_rest',    name='read_rest'
    )
    read_rest.setVolume(1.0)
    
    # --- Initialize components for Routine "_et_instruct" ---
    text_et = visual.TextStim(win=win, name='text_et',
        text='During the calibration, you will see a target circle moving around the screen. Please try to track it with your eyes.\n\nMake sure to keep looking at the circle when it stops, and follow it when it moves. It is important that you keep your head on the chin rest once this part begins.\n\n\nPress any of the white keys when you are ready, and our team will start the calibration for you',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_et = keyboard.Keyboard(deviceName='key_et')
    read_et = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_et',    name='read_et'
    )
    read_et.setVolume(1.0)
    
    # --- Initialize components for Routine "_et_mask" ---
    text_mask = visual.TextStim(win=win, name='text_mask',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "instruct_continue" ---
    text_continue = visual.TextStim(win=win, name='text_continue',
        text='We will now continue with the task.\n\nRemember: if an image matches the target, press the Green key, and if it does not match the target, press the Red key.\n\nPlease respond as quickly and accurately as possible after you see each image.\n\n\nPress the green key to begin',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_continue = keyboard.Keyboard(deviceName='key_continue')
    read_continue = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_continue',    name='read_continue'
    )
    read_continue.setVolume(1.0)
    
    # --- Initialize components for Routine "__end__" ---
    text_thank_you = visual.TextStim(win=win, name='text_thank_you',
        text='Thank you. You have completed this task!',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    read_thank_you = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_thank_you',    name='read_thank_you'
    )
    read_thank_you.setVolume(1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "_welcome" ---
    # create an object to store info about Routine _welcome
    _welcome = data.Routine(
        name='_welcome',
        components=[text_welcome, key_welcome, read_welcome],
    )
    _welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_welcome
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    read_welcome.setSound('resource/welcome.wav', hamming=True)
    read_welcome.setVolume(1.0, log=False)
    read_welcome.seek(0)
    # store start times for _welcome
    _welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _welcome.tStart = globalClock.getTime(format='float')
    _welcome.status = STARTED
    _welcome.maxDuration = None
    # keep track of which components have finished
    _welcomeComponents = _welcome.components
    for thisComponent in _welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_welcome" ---
    _welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome* updates
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_welcome.status = STARTED
            text_welcome.setAutoDraw(True)
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            # update params
            pass
        
        # *key_welcome* updates
        waitOnFlip = False
        
        # if key_welcome is starting this frame...
        if key_welcome.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_welcome.frameNStart = frameN  # exact frame index
            key_welcome.tStart = t  # local t and not account for scr refresh
            key_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_welcome_allKeys.extend(theseKeys)
            if len(_key_welcome_allKeys):
                key_welcome.keys = _key_welcome_allKeys[-1].name  # just the last key pressed
                key_welcome.rt = _key_welcome_allKeys[-1].rt
                key_welcome.duration = _key_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_welcome* updates
        
        # if read_welcome is starting this frame...
        if read_welcome.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_welcome.frameNStart = frameN  # exact frame index
            read_welcome.tStart = t  # local t and not account for scr refresh
            read_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_welcome.status = STARTED
            read_welcome.play(when=win)  # sync with win flip
        
        # if read_welcome is stopping this frame...
        if read_welcome.status == STARTED:
            if bool(False) or read_welcome.isFinished:
                # keep track of stop time/frame for later
                read_welcome.tStop = t  # not accounting for scr refresh
                read_welcome.tStopRefresh = tThisFlipGlobal  # on global time
                read_welcome.frameNStop = frameN  # exact frame index
                # update status
                read_welcome.status = FINISHED
                read_welcome.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_welcome]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_welcome" ---
    for thisComponent in _welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _welcome
    _welcome.tStop = globalClock.getTime(format='float')
    _welcome.tStopRefresh = tThisFlipGlobal
    read_welcome.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_et_instruct" ---
    # create an object to store info about Routine _et_instruct
    _et_instruct = data.Routine(
        name='_et_instruct',
        components=[text_et, key_et, read_et],
    )
    _et_instruct.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_et
    key_et.keys = []
    key_et.rt = []
    _key_et_allKeys = []
    read_et.setSound('resource/eyetrack_calibrate_instruct.wav', hamming=True)
    read_et.setVolume(1.0, log=False)
    read_et.seek(0)
    # store start times for _et_instruct
    _et_instruct.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _et_instruct.tStart = globalClock.getTime(format='float')
    _et_instruct.status = STARTED
    _et_instruct.maxDuration = None
    # keep track of which components have finished
    _et_instructComponents = _et_instruct.components
    for thisComponent in _et_instruct.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_et_instruct" ---
    _et_instruct.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_et* updates
        
        # if text_et is starting this frame...
        if text_et.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_et.frameNStart = frameN  # exact frame index
            text_et.tStart = t  # local t and not account for scr refresh
            text_et.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_et, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_et.status = STARTED
            text_et.setAutoDraw(True)
        
        # if text_et is active this frame...
        if text_et.status == STARTED:
            # update params
            pass
        
        # *key_et* updates
        waitOnFlip = False
        
        # if key_et is starting this frame...
        if key_et.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_et.frameNStart = frameN  # exact frame index
            key_et.tStart = t  # local t and not account for scr refresh
            key_et.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_et, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_et.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_et.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_et.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_et.status == STARTED and not waitOnFlip:
            theseKeys = key_et.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_et_allKeys.extend(theseKeys)
            if len(_key_et_allKeys):
                key_et.keys = _key_et_allKeys[-1].name  # just the last key pressed
                key_et.rt = _key_et_allKeys[-1].rt
                key_et.duration = _key_et_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_et* updates
        
        # if read_et is starting this frame...
        if read_et.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_et.frameNStart = frameN  # exact frame index
            read_et.tStart = t  # local t and not account for scr refresh
            read_et.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_et.status = STARTED
            read_et.play(when=win)  # sync with win flip
        
        # if read_et is stopping this frame...
        if read_et.status == STARTED:
            if bool(False) or read_et.isFinished:
                # keep track of stop time/frame for later
                read_et.tStop = t  # not accounting for scr refresh
                read_et.tStopRefresh = tThisFlipGlobal  # on global time
                read_et.frameNStop = frameN  # exact frame index
                # update status
                read_et.status = FINISHED
                read_et.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_et]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _et_instruct.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _et_instruct.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_et_instruct" ---
    for thisComponent in _et_instruct.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _et_instruct
    _et_instruct.tStop = globalClock.getTime(format='float')
    _et_instruct.tStopRefresh = tThisFlipGlobal
    read_et.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "_et_instruct" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_et_mask" ---
    # create an object to store info about Routine _et_mask
    _et_mask = data.Routine(
        name='_et_mask',
        components=[text_mask],
    )
    _et_mask.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for _et_mask
    _et_mask.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _et_mask.tStart = globalClock.getTime(format='float')
    _et_mask.status = STARTED
    _et_mask.maxDuration = None
    # keep track of which components have finished
    _et_maskComponents = _et_mask.components
    for thisComponent in _et_mask.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_et_mask" ---
    _et_mask.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.05:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_mask* updates
        
        # if text_mask is starting this frame...
        if text_mask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_mask.frameNStart = frameN  # exact frame index
            text_mask.tStart = t  # local t and not account for scr refresh
            text_mask.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_mask, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_mask.status = STARTED
            text_mask.setAutoDraw(True)
        
        # if text_mask is active this frame...
        if text_mask.status == STARTED:
            # update params
            pass
        
        # if text_mask is stopping this frame...
        if text_mask.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_mask.tStartRefresh + 0.05-frameTolerance:
                # keep track of stop time/frame for later
                text_mask.tStop = t  # not accounting for scr refresh
                text_mask.tStopRefresh = tThisFlipGlobal  # on global time
                text_mask.frameNStop = frameN  # exact frame index
                # update status
                text_mask.status = FINISHED
                text_mask.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _et_mask.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _et_mask.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_et_mask" ---
    for thisComponent in _et_mask.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _et_mask
    _et_mask.tStop = globalClock.getTime(format='float')
    _et_mask.tStopRefresh = tThisFlipGlobal
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if _et_mask.maxDurationReached:
        routineTimer.addTime(-_et_mask.maxDuration)
    elif _et_mask.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.050000)
    thisExp.nextEntry()
    # define target for _et_cal
    _et_calTarget = visual.TargetStim(win, 
        name='_et_calTarget',
        radius=0.015, fillColor='white', borderColor='green', lineWidth=2.0,
        innerRadius=0.005, innerFillColor='black', innerBorderColor='black', innerLineWidth=2.0,
        colorSpace='rgb', units=None
    )
    # define parameters for _et_cal
    _et_cal = hardware.eyetracker.EyetrackerCalibration(win, 
        eyetracker, _et_calTarget,
        units=None, colorSpace='rgb',
        progressMode='time', targetDur=1.5, expandScale=1.5,
        targetLayout='NINE_POINTS', randomisePos=True, textColor='white',
        movementAnimation=True, targetDelay=1.0
    )
    # run calibration
    _et_cal.run()
    # clear any keypresses from during _et_cal so they don't interfere with the experiment
    defaultKeyboard.clearEvents()
    thisExp.nextEntry()
    # the Routine "_et_cal" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "__start__" ---
    # create an object to store info about Routine __start__
    __start__ = data.Routine(
        name='__start__',
        components=[text_start, read_start, etRecord],
    )
    __start__.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    read_start.setSound('resource/ready_to_begin.wav', secs=1.8, hamming=True)
    read_start.setVolume(1.0, log=False)
    read_start.seek(0)
    # store start times for __start__
    __start__.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    __start__.tStart = globalClock.getTime(format='float')
    __start__.status = STARTED
    __start__.maxDuration = None
    # keep track of which components have finished
    __start__Components = __start__.components
    for thisComponent in __start__.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__start__" ---
    __start__.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_start* updates
        
        # if text_start is starting this frame...
        if text_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_start.frameNStart = frameN  # exact frame index
            text_start.tStart = t  # local t and not account for scr refresh
            text_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_start.status = STARTED
            text_start.setAutoDraw(True)
        
        # if text_start is active this frame...
        if text_start.status == STARTED:
            # update params
            pass
        
        # if text_start is stopping this frame...
        if text_start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_start.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                text_start.tStop = t  # not accounting for scr refresh
                text_start.tStopRefresh = tThisFlipGlobal  # on global time
                text_start.frameNStop = frameN  # exact frame index
                # update status
                text_start.status = FINISHED
                text_start.setAutoDraw(False)
        
        # *read_start* updates
        
        # if read_start is starting this frame...
        if read_start.status == NOT_STARTED and t >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            read_start.frameNStart = frameN  # exact frame index
            read_start.tStart = t  # local t and not account for scr refresh
            read_start.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_start.status = STARTED
            read_start.play()  # start the sound (it finishes automatically)
        
        # if read_start is stopping this frame...
        if read_start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > read_start.tStartRefresh + 1.8-frameTolerance or read_start.isFinished:
                # keep track of stop time/frame for later
                read_start.tStop = t  # not accounting for scr refresh
                read_start.tStopRefresh = tThisFlipGlobal  # on global time
                read_start.frameNStop = frameN  # exact frame index
                # update status
                read_start.status = FINISHED
                read_start.stop()
        
        # *etRecord* updates
        
        # if etRecord is starting this frame...
        if etRecord.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord.frameNStart = frameN  # exact frame index
            etRecord.tStart = t  # local t and not account for scr refresh
            etRecord.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('etRecord.started', t)
            # update status
            etRecord.status = STARTED
            etRecord.start()
        if etRecord.status == STARTED:
            etRecord.tStop = t  # not accounting for scr refresh
            etRecord.tStopRefresh = tThisFlipGlobal  # on global time
            etRecord.frameNStop = frameN  # exact frame index
            etRecord.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_start]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            __start__.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __start__.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__start__" ---
    for thisComponent in __start__.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for __start__
    __start__.tStop = globalClock.getTime(format='float')
    __start__.tStopRefresh = tThisFlipGlobal
    read_start.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if __start__.maxDurationReached:
        routineTimer.addTime(-__start__.maxDuration)
    elif __start__.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instruct" ---
    # create an object to store info about Routine instruct
    instruct = data.Routine(
        name='instruct',
        components=[text_instruct, key_instruct, read_instruct],
    )
    instruct.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct
    key_instruct.keys = []
    key_instruct.rt = []
    _key_instruct_allKeys = []
    read_instruct.setSound('resource/instruct.wav', hamming=True)
    read_instruct.setVolume(1.0, log=False)
    read_instruct.seek(0)
    # store start times for instruct
    instruct.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct.tStart = globalClock.getTime(format='float')
    instruct.status = STARTED
    instruct.maxDuration = None
    # keep track of which components have finished
    instructComponents = instruct.components
    for thisComponent in instruct.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct" ---
    instruct.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct* updates
        
        # if text_instruct is starting this frame...
        if text_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct.frameNStart = frameN  # exact frame index
            text_instruct.tStart = t  # local t and not account for scr refresh
            text_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct.status = STARTED
            text_instruct.setAutoDraw(True)
        
        # if text_instruct is active this frame...
        if text_instruct.status == STARTED:
            # update params
            pass
        
        # *key_instruct* updates
        waitOnFlip = False
        
        # if key_instruct is starting this frame...
        if key_instruct.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct.frameNStart = frameN  # exact frame index
            key_instruct.tStart = t  # local t and not account for scr refresh
            key_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_allKeys.extend(theseKeys)
            if len(_key_instruct_allKeys):
                key_instruct.keys = _key_instruct_allKeys[-1].name  # just the last key pressed
                key_instruct.rt = _key_instruct_allKeys[-1].rt
                key_instruct.duration = _key_instruct_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct* updates
        
        # if read_instruct is starting this frame...
        if read_instruct.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct.frameNStart = frameN  # exact frame index
            read_instruct.tStart = t  # local t and not account for scr refresh
            read_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct.status = STARTED
            read_instruct.play(when=win)  # sync with win flip
        
        # if read_instruct is stopping this frame...
        if read_instruct.status == STARTED:
            if bool(False) or read_instruct.isFinished:
                # keep track of stop time/frame for later
                read_instruct.tStop = t  # not accounting for scr refresh
                read_instruct.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct.frameNStop = frameN  # exact frame index
                # update status
                read_instruct.status = FINISHED
                read_instruct.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct" ---
    for thisComponent in instruct.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct
    instruct.tStop = globalClock.getTime(format='float')
    instruct.tStopRefresh = tThisFlipGlobal
    read_instruct.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_diagram" ---
    # create an object to store info about Routine instruct_diagram
    instruct_diagram = data.Routine(
        name='instruct_diagram',
        components=[image_schematic, text_schematic, key_schematic, read_schematic],
    )
    instruct_diagram.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from adjust_schematic_size
    scale_to_size(image_schematic, 1.5)
    
    # create starting attributes for key_schematic
    key_schematic.keys = []
    key_schematic.rt = []
    _key_schematic_allKeys = []
    read_schematic.setSound('resource/instruct_diagram.wav', hamming=True)
    read_schematic.setVolume(1.0, log=False)
    read_schematic.seek(0)
    # store start times for instruct_diagram
    instruct_diagram.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_diagram.tStart = globalClock.getTime(format='float')
    instruct_diagram.status = STARTED
    instruct_diagram.maxDuration = None
    # keep track of which components have finished
    instruct_diagramComponents = instruct_diagram.components
    for thisComponent in instruct_diagram.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_diagram" ---
    instruct_diagram.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_schematic* updates
        
        # if image_schematic is starting this frame...
        if image_schematic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_schematic.frameNStart = frameN  # exact frame index
            image_schematic.tStart = t  # local t and not account for scr refresh
            image_schematic.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_schematic, 'tStartRefresh')  # time at next scr refresh
            # update status
            image_schematic.status = STARTED
            image_schematic.setAutoDraw(True)
        
        # if image_schematic is active this frame...
        if image_schematic.status == STARTED:
            # update params
            pass
        
        # *text_schematic* updates
        
        # if text_schematic is starting this frame...
        if text_schematic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_schematic.frameNStart = frameN  # exact frame index
            text_schematic.tStart = t  # local t and not account for scr refresh
            text_schematic.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_schematic, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_schematic.status = STARTED
            text_schematic.setAutoDraw(True)
        
        # if text_schematic is active this frame...
        if text_schematic.status == STARTED:
            # update params
            pass
        
        # *key_schematic* updates
        waitOnFlip = False
        
        # if key_schematic is starting this frame...
        if key_schematic.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_schematic.frameNStart = frameN  # exact frame index
            key_schematic.tStart = t  # local t and not account for scr refresh
            key_schematic.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_schematic, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_schematic.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_schematic.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_schematic.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_schematic.status == STARTED and not waitOnFlip:
            theseKeys = key_schematic.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_schematic_allKeys.extend(theseKeys)
            if len(_key_schematic_allKeys):
                key_schematic.keys = _key_schematic_allKeys[-1].name  # just the last key pressed
                key_schematic.rt = _key_schematic_allKeys[-1].rt
                key_schematic.duration = _key_schematic_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_schematic* updates
        
        # if read_schematic is starting this frame...
        if read_schematic.status == NOT_STARTED and tThisFlip >= 4.0-frameTolerance:
            # keep track of start time/frame for later
            read_schematic.frameNStart = frameN  # exact frame index
            read_schematic.tStart = t  # local t and not account for scr refresh
            read_schematic.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_schematic.status = STARTED
            read_schematic.play(when=win)  # sync with win flip
        
        # if read_schematic is stopping this frame...
        if read_schematic.status == STARTED:
            if bool(False) or read_schematic.isFinished:
                # keep track of stop time/frame for later
                read_schematic.tStop = t  # not accounting for scr refresh
                read_schematic.tStopRefresh = tThisFlipGlobal  # on global time
                read_schematic.frameNStop = frameN  # exact frame index
                # update status
                read_schematic.status = FINISHED
                read_schematic.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_schematic]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_diagram.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_diagram.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_diagram" ---
    for thisComponent in instruct_diagram.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_diagram
    instruct_diagram.tStop = globalClock.getTime(format='float')
    instruct_diagram.tStopRefresh = tThisFlipGlobal
    read_schematic.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_diagram" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_practice" ---
    # create an object to store info about Routine instruct_practice
    instruct_practice = data.Routine(
        name='instruct_practice',
        components=[text_practice, key_practice, read_practice],
    )
    instruct_practice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_practice
    key_practice.keys = []
    key_practice.rt = []
    _key_practice_allKeys = []
    read_practice.setSound('resource/instruct_practice.wav', hamming=True)
    read_practice.setVolume(1.0, log=False)
    read_practice.seek(0)
    # store start times for instruct_practice
    instruct_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_practice.tStart = globalClock.getTime(format='float')
    instruct_practice.status = STARTED
    instruct_practice.maxDuration = None
    # keep track of which components have finished
    instruct_practiceComponents = instruct_practice.components
    for thisComponent in instruct_practice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_practice" ---
    instruct_practice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_practice* updates
        
        # if text_practice is starting this frame...
        if text_practice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_practice.frameNStart = frameN  # exact frame index
            text_practice.tStart = t  # local t and not account for scr refresh
            text_practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_practice, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_practice.status = STARTED
            text_practice.setAutoDraw(True)
        
        # if text_practice is active this frame...
        if text_practice.status == STARTED:
            # update params
            pass
        
        # *key_practice* updates
        waitOnFlip = False
        
        # if key_practice is starting this frame...
        if key_practice.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_practice.frameNStart = frameN  # exact frame index
            key_practice.tStart = t  # local t and not account for scr refresh
            key_practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_practice, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_practice.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_practice.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_practice.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_practice.status == STARTED and not waitOnFlip:
            theseKeys = key_practice.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
            _key_practice_allKeys.extend(theseKeys)
            if len(_key_practice_allKeys):
                key_practice.keys = _key_practice_allKeys[-1].name  # just the last key pressed
                key_practice.rt = _key_practice_allKeys[-1].rt
                key_practice.duration = _key_practice_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_practice* updates
        
        # if read_practice is starting this frame...
        if read_practice.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_practice.frameNStart = frameN  # exact frame index
            read_practice.tStart = t  # local t and not account for scr refresh
            read_practice.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_practice.status = STARTED
            read_practice.play(when=win)  # sync with win flip
        
        # if read_practice is stopping this frame...
        if read_practice.status == STARTED:
            if bool(False) or read_practice.isFinished:
                # keep track of stop time/frame for later
                read_practice.tStop = t  # not accounting for scr refresh
                read_practice.tStopRefresh = tThisFlipGlobal  # on global time
                read_practice.frameNStop = frameN  # exact frame index
                # update status
                read_practice.status = FINISHED
                read_practice.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_practice]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_practice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_practice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_practice" ---
    for thisComponent in instruct_practice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_practice
    instruct_practice.tStop = globalClock.getTime(format='float')
    instruct_practice.tStopRefresh = tThisFlipGlobal
    read_practice.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler2(
        name='practice_loop',
        nReps=99.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "instruct_practice_repeat" ---
        # create an object to store info about Routine instruct_practice_repeat
        instruct_practice_repeat = data.Routine(
            name='instruct_practice_repeat',
            components=[test_practice_repeat, key_practice_repeat, read_practice_repeat],
        )
        instruct_practice_repeat.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_practice_repeat
        key_practice_repeat.keys = []
        key_practice_repeat.rt = []
        _key_practice_repeat_allKeys = []
        read_practice_repeat.setSound('resource/instruct_practice_repeat.wav', hamming=True)
        read_practice_repeat.setVolume(1.0, log=False)
        read_practice_repeat.seek(0)
        # Run 'Begin Routine' code from skip_routine_check
        # Start a practice block
        dev.activate_line(bitmask=block_start_code)
        eyetracker.sendMessage(block_start_code)
        core.wait(0.5)  # wait 500ms before trial triggers
        
        # Skip this routine if first time doing practice
        if practice_loop.thisRepN == 0:
            continueRoutine = False
        
        # store start times for instruct_practice_repeat
        instruct_practice_repeat.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instruct_practice_repeat.tStart = globalClock.getTime(format='float')
        instruct_practice_repeat.status = STARTED
        instruct_practice_repeat.maxDuration = None
        # keep track of which components have finished
        instruct_practice_repeatComponents = instruct_practice_repeat.components
        for thisComponent in instruct_practice_repeat.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instruct_practice_repeat" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        instruct_practice_repeat.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *test_practice_repeat* updates
            
            # if test_practice_repeat is starting this frame...
            if test_practice_repeat.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_practice_repeat.frameNStart = frameN  # exact frame index
                test_practice_repeat.tStart = t  # local t and not account for scr refresh
                test_practice_repeat.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_practice_repeat, 'tStartRefresh')  # time at next scr refresh
                # update status
                test_practice_repeat.status = STARTED
                test_practice_repeat.setAutoDraw(True)
            
            # if test_practice_repeat is active this frame...
            if test_practice_repeat.status == STARTED:
                # update params
                pass
            
            # *key_practice_repeat* updates
            waitOnFlip = False
            
            # if key_practice_repeat is starting this frame...
            if key_practice_repeat.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_practice_repeat.frameNStart = frameN  # exact frame index
                key_practice_repeat.tStart = t  # local t and not account for scr refresh
                key_practice_repeat.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_practice_repeat, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_practice_repeat.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_practice_repeat.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_practice_repeat.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_practice_repeat.status == STARTED and not waitOnFlip:
                theseKeys = key_practice_repeat.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
                _key_practice_repeat_allKeys.extend(theseKeys)
                if len(_key_practice_repeat_allKeys):
                    key_practice_repeat.keys = _key_practice_repeat_allKeys[-1].name  # just the last key pressed
                    key_practice_repeat.rt = _key_practice_repeat_allKeys[-1].rt
                    key_practice_repeat.duration = _key_practice_repeat_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *read_practice_repeat* updates
            
            # if read_practice_repeat is starting this frame...
            if read_practice_repeat.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                read_practice_repeat.frameNStart = frameN  # exact frame index
                read_practice_repeat.tStart = t  # local t and not account for scr refresh
                read_practice_repeat.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                read_practice_repeat.status = STARTED
                read_practice_repeat.play(when=win)  # sync with win flip
            
            # if read_practice_repeat is stopping this frame...
            if read_practice_repeat.status == STARTED:
                if bool(False) or read_practice_repeat.isFinished:
                    # keep track of stop time/frame for later
                    read_practice_repeat.tStop = t  # not accounting for scr refresh
                    read_practice_repeat.tStopRefresh = tThisFlipGlobal  # on global time
                    read_practice_repeat.frameNStop = frameN  # exact frame index
                    # update status
                    read_practice_repeat.status = FINISHED
                    read_practice_repeat.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[read_practice_repeat]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instruct_practice_repeat.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruct_practice_repeat.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruct_practice_repeat" ---
        for thisComponent in instruct_practice_repeat.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instruct_practice_repeat
        instruct_practice_repeat.tStop = globalClock.getTime(format='float')
        instruct_practice_repeat.tStopRefresh = tThisFlipGlobal
        read_practice_repeat.pause()  # ensure sound has stopped at end of Routine
        # the Routine "instruct_practice_repeat" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        practice_trials = data.TrialHandler2(
            name='practice_trials',
            nReps=n_trials_practice, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(practice_trials)  # add the loop to the experiment
        thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPractice_trial in practice_trials:
            currentLoop = practice_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
            if thisPractice_trial != None:
                for paramName in thisPractice_trial:
                    globals()[paramName] = thisPractice_trial[paramName]
            
            # --- Prepare to start Routine "practice_setup" ---
            # create an object to store info about Routine practice_setup
            practice_setup = data.Routine(
                name='practice_setup',
                components=[],
            )
            practice_setup.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from setup_practice_trial
            # Set up the content for a practice trial
            image_fn = image_fn_practice_list[practice_trials.thisRepN]
            imageType = imageType_practice_list[practice_trials.thisRepN]
            correct_resps = ['1' if itype == 1 else '2' for itype in imageType]
            
            # store start times for practice_setup
            practice_setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            practice_setup.tStart = globalClock.getTime(format='float')
            practice_setup.status = STARTED
            practice_setup.maxDuration = None
            # keep track of which components have finished
            practice_setupComponents = practice_setup.components
            for thisComponent in practice_setup.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "practice_setup" ---
            # if trial has changed, end Routine now
            if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
                continueRoutine = False
            practice_setup.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    practice_setup.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in practice_setup.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "practice_setup" ---
            for thisComponent in practice_setup.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for practice_setup
            practice_setup.tStop = globalClock.getTime(format='float')
            practice_setup.tStopRefresh = tThisFlipGlobal
            # Run 'End Routine' code from setup_practice_trial
            thisExp.addData('target_image', image_fn[1])  # adding Target Image Name to .csv file
            thisExp.addData('distractor_image', image_fn[0])  # adding Distractor Image Name to .csv file
            
            # the Routine "practice_setup" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial_target" ---
            # create an object to store info about Routine trial_target
            trial_target = data.Routine(
                name='trial_target',
                components=[green_square_part1, green_square_part2, image_target, key_resp_catch_delay, text_fixation_first],
            )
            trial_target.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_target.setImage(image_fn[1])
            # Run 'Begin Routine' code from adjust_image_target_size
            box_size = 0.395
            scale_to_size(image_target, box_size)
            
            # create starting attributes for key_resp_catch_delay
            key_resp_catch_delay.keys = []
            key_resp_catch_delay.rt = []
            _key_resp_catch_delay_allKeys = []
            # Run 'Begin Routine' code from initiate_trial_sequence
            # Set the first inter-stimulus interval (ISI) of fixation
            isi = rng.uniform(1.1, 1.4)
            
            # Begin a fresh counter for the number of images
            thisImageNumber = 0
            
            # Run 'Begin Routine' code from trigger_target
            target_started = False
            
            # store start times for trial_target
            trial_target.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_target.tStart = globalClock.getTime(format='float')
            trial_target.status = STARTED
            thisExp.addData('trial_target.started', trial_target.tStart)
            trial_target.maxDuration = None
            # keep track of which components have finished
            trial_targetComponents = trial_target.components
            for thisComponent in trial_target.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_target" ---
            # if trial has changed, end Routine now
            if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
                continueRoutine = False
            trial_target.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *green_square_part1* updates
                
                # if green_square_part1 is starting this frame...
                if green_square_part1.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                    # keep track of start time/frame for later
                    green_square_part1.frameNStart = frameN  # exact frame index
                    green_square_part1.tStart = t  # local t and not account for scr refresh
                    green_square_part1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(green_square_part1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_square_part1.started')
                    # update status
                    green_square_part1.status = STARTED
                    green_square_part1.setAutoDraw(True)
                
                # if green_square_part1 is active this frame...
                if green_square_part1.status == STARTED:
                    # update params
                    pass
                
                # if green_square_part1 is stopping this frame...
                if green_square_part1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > green_square_part1.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        green_square_part1.tStop = t  # not accounting for scr refresh
                        green_square_part1.tStopRefresh = tThisFlipGlobal  # on global time
                        green_square_part1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'green_square_part1.stopped')
                        # update status
                        green_square_part1.status = FINISHED
                        green_square_part1.setAutoDraw(False)
                
                # *green_square_part2* updates
                
                # if green_square_part2 is starting this frame...
                if green_square_part2.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                    # keep track of start time/frame for later
                    green_square_part2.frameNStart = frameN  # exact frame index
                    green_square_part2.tStart = t  # local t and not account for scr refresh
                    green_square_part2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(green_square_part2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_square_part2.started')
                    # update status
                    green_square_part2.status = STARTED
                    green_square_part2.setAutoDraw(True)
                
                # if green_square_part2 is active this frame...
                if green_square_part2.status == STARTED:
                    # update params
                    pass
                
                # if green_square_part2 is stopping this frame...
                if green_square_part2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > green_square_part2.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        green_square_part2.tStop = t  # not accounting for scr refresh
                        green_square_part2.tStopRefresh = tThisFlipGlobal  # on global time
                        green_square_part2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'green_square_part2.stopped')
                        # update status
                        green_square_part2.status = FINISHED
                        green_square_part2.setAutoDraw(False)
                
                # *image_target* updates
                
                # if image_target is starting this frame...
                if image_target.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                    # keep track of start time/frame for later
                    image_target.frameNStart = frameN  # exact frame index
                    image_target.tStart = t  # local t and not account for scr refresh
                    image_target.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_target, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_target.started')
                    # update status
                    image_target.status = STARTED
                    image_target.setAutoDraw(True)
                
                # if image_target is active this frame...
                if image_target.status == STARTED:
                    # update params
                    pass
                
                # if image_target is stopping this frame...
                if image_target.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_target.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_target.tStop = t  # not accounting for scr refresh
                        image_target.tStopRefresh = tThisFlipGlobal  # on global time
                        image_target.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_target.stopped')
                        # update status
                        image_target.status = FINISHED
                        image_target.setAutoDraw(False)
                
                # *key_resp_catch_delay* updates
                waitOnFlip = False
                
                # if key_resp_catch_delay is starting this frame...
                if key_resp_catch_delay.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_catch_delay.frameNStart = frameN  # exact frame index
                    key_resp_catch_delay.tStart = t  # local t and not account for scr refresh
                    key_resp_catch_delay.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_catch_delay, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_catch_delay.started')
                    # update status
                    key_resp_catch_delay.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_catch_delay.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_catch_delay.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_catch_delay is stopping this frame...
                if key_resp_catch_delay.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_catch_delay.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_catch_delay.tStop = t  # not accounting for scr refresh
                        key_resp_catch_delay.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_catch_delay.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_catch_delay.stopped')
                        # update status
                        key_resp_catch_delay.status = FINISHED
                        key_resp_catch_delay.status = FINISHED
                if key_resp_catch_delay.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_catch_delay.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_catch_delay_allKeys.extend(theseKeys)
                    if len(_key_resp_catch_delay_allKeys):
                        key_resp_catch_delay.keys = [key.name for key in _key_resp_catch_delay_allKeys]  # storing all keys
                        key_resp_catch_delay.rt = [key.rt for key in _key_resp_catch_delay_allKeys]
                        key_resp_catch_delay.duration = [key.duration for key in _key_resp_catch_delay_allKeys]
                
                # *text_fixation_first* updates
                
                # if text_fixation_first is starting this frame...
                if text_fixation_first.status == NOT_STARTED and image_target.status == FINISHED:
                    # keep track of start time/frame for later
                    text_fixation_first.frameNStart = frameN  # exact frame index
                    text_fixation_first.tStart = t  # local t and not account for scr refresh
                    text_fixation_first.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_fixation_first, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_fixation_first.started')
                    # update status
                    text_fixation_first.status = STARTED
                    text_fixation_first.setAutoDraw(True)
                
                # if text_fixation_first is active this frame...
                if text_fixation_first.status == STARTED:
                    # update params
                    pass
                
                # if text_fixation_first is stopping this frame...
                if text_fixation_first.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_fixation_first.tStartRefresh + isi-frameTolerance:
                        # keep track of stop time/frame for later
                        text_fixation_first.tStop = t  # not accounting for scr refresh
                        text_fixation_first.tStopRefresh = tThisFlipGlobal  # on global time
                        text_fixation_first.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_fixation_first.stopped')
                        # update status
                        text_fixation_first.status = FINISHED
                        text_fixation_first.setAutoDraw(False)
                # Run 'Each Frame' code from trigger_target
                if image_target.status == STARTED and not target_started:
                    win.callOnFlip(dev.activate_line, bitmask=target_start_code)
                    win.callOnFlip(eyetracker.sendMessage, target_start_code)
                    target_started = True
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial_target.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_target.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_target" ---
            for thisComponent in trial_target.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_target
            trial_target.tStop = globalClock.getTime(format='float')
            trial_target.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_target.stopped', trial_target.tStop)
            # check responses
            if key_resp_catch_delay.keys in ['', [], None]:  # No response was made
                key_resp_catch_delay.keys = None
            practice_trials.addData('key_resp_catch_delay.keys',key_resp_catch_delay.keys)
            if key_resp_catch_delay.keys != None:  # we had a response
                practice_trials.addData('key_resp_catch_delay.rt', key_resp_catch_delay.rt)
                practice_trials.addData('key_resp_catch_delay.duration', key_resp_catch_delay.duration)
            # the Routine "trial_target" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            practice_sequence = data.TrialHandler2(
                name='practice_sequence',
                nReps=n_images_per_trial, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(practice_sequence)  # add the loop to the experiment
            thisPractice_sequence = practice_sequence.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisPractice_sequence.rgb)
            if thisPractice_sequence != None:
                for paramName in thisPractice_sequence:
                    globals()[paramName] = thisPractice_sequence[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisPractice_sequence in practice_sequence:
                currentLoop = practice_sequence
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisPractice_sequence.rgb)
                if thisPractice_sequence != None:
                    for paramName in thisPractice_sequence:
                        globals()[paramName] = thisPractice_sequence[paramName]
                
                # --- Prepare to start Routine "trial_image" ---
                # create an object to store info about Routine trial_image
                trial_image = data.Routine(
                    name='trial_image',
                    components=[white_square, image_test, text_fixation, key_resp],
                )
                trial_image.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from setup_test_image
                # Obtain the current image filename
                thisImageType = imageType[thisImageNumber]
                image_test_fn = image_fn[thisImageType]
                thisCorrectResp = correct_resps[thisImageNumber]
                
                # Set the inter-stimulus interval (ISI) of fixation
                if thisImageNumber < (n_images_per_trial - 1):
                    isi = rng.uniform(1.1, 1.4)
                else:
                    # skip showing a fixation cross after the last image in sequence
                    text_fixation.status = FINISHED
                    # but still collect behavioral responses for another 1.5s
                    isi = 1.5
                
                image_test.setImage(image_test_fn)
                # Run 'Begin Routine' code from adjust_image_test_size
                scale_to_size(image_test, box_size)
                
                # create starting attributes for key_resp
                key_resp.keys = []
                key_resp.rt = []
                _key_resp_allKeys = []
                # Run 'Begin Routine' code from trigger_image
                test_started = False
                
                # store start times for trial_image
                trial_image.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                trial_image.tStart = globalClock.getTime(format='float')
                trial_image.status = STARTED
                thisExp.addData('trial_image.started', trial_image.tStart)
                trial_image.maxDuration = None
                # keep track of which components have finished
                trial_imageComponents = trial_image.components
                for thisComponent in trial_image.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "trial_image" ---
                # if trial has changed, end Routine now
                if isinstance(practice_sequence, data.TrialHandler2) and thisPractice_sequence.thisN != practice_sequence.thisTrial.thisN:
                    continueRoutine = False
                trial_image.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *white_square* updates
                    
                    # if white_square is starting this frame...
                    if white_square.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        white_square.frameNStart = frameN  # exact frame index
                        white_square.tStart = t  # local t and not account for scr refresh
                        white_square.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(white_square, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'white_square.started')
                        # update status
                        white_square.status = STARTED
                        white_square.setAutoDraw(True)
                    
                    # if white_square is active this frame...
                    if white_square.status == STARTED:
                        # update params
                        pass
                    
                    # if white_square is stopping this frame...
                    if white_square.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > white_square.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            white_square.tStop = t  # not accounting for scr refresh
                            white_square.tStopRefresh = tThisFlipGlobal  # on global time
                            white_square.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'white_square.stopped')
                            # update status
                            white_square.status = FINISHED
                            white_square.setAutoDraw(False)
                    
                    # *image_test* updates
                    
                    # if image_test is starting this frame...
                    if image_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_test.frameNStart = frameN  # exact frame index
                        image_test.tStart = t  # local t and not account for scr refresh
                        image_test.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_test, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_test.started')
                        # update status
                        image_test.status = STARTED
                        image_test.setAutoDraw(True)
                    
                    # if image_test is active this frame...
                    if image_test.status == STARTED:
                        # update params
                        pass
                    
                    # if image_test is stopping this frame...
                    if image_test.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_test.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            image_test.tStop = t  # not accounting for scr refresh
                            image_test.tStopRefresh = tThisFlipGlobal  # on global time
                            image_test.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_test.stopped')
                            # update status
                            image_test.status = FINISHED
                            image_test.setAutoDraw(False)
                    
                    # *text_fixation* updates
                    
                    # if text_fixation is starting this frame...
                    if text_fixation.status == NOT_STARTED and image_test.status == FINISHED:
                        # keep track of start time/frame for later
                        text_fixation.frameNStart = frameN  # exact frame index
                        text_fixation.tStart = t  # local t and not account for scr refresh
                        text_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_fixation, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_fixation.started')
                        # update status
                        text_fixation.status = STARTED
                        text_fixation.setAutoDraw(True)
                    
                    # if text_fixation is active this frame...
                    if text_fixation.status == STARTED:
                        # update params
                        pass
                    
                    # if text_fixation is stopping this frame...
                    if text_fixation.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_fixation.tStartRefresh + isi-frameTolerance:
                            # keep track of stop time/frame for later
                            text_fixation.tStop = t  # not accounting for scr refresh
                            text_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                            text_fixation.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_fixation.stopped')
                            # update status
                            text_fixation.status = FINISHED
                            text_fixation.setAutoDraw(False)
                    
                    # *key_resp* updates
                    waitOnFlip = False
                    
                    # if key_resp is starting this frame...
                    if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp.frameNStart = frameN  # exact frame index
                        key_resp.tStart = t  # local t and not account for scr refresh
                        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.started')
                        # update status
                        key_resp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_resp is stopping this frame...
                    if key_resp.status == STARTED:
                        # is it time to stop? (based on local clock)
                        if tThisFlip > isi + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp.tStop = t  # not accounting for scr refresh
                            key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_resp.stopped')
                            # update status
                            key_resp.status = FINISHED
                            key_resp.status = FINISHED
                    if key_resp.status == STARTED and not waitOnFlip:
                        theseKeys = key_resp.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_allKeys.extend(theseKeys)
                        if len(_key_resp_allKeys):
                            key_resp.keys = [key.name for key in _key_resp_allKeys]  # storing all keys
                            key_resp.rt = [key.rt for key in _key_resp_allKeys]
                            key_resp.duration = [key.duration for key in _key_resp_allKeys]
                    # Run 'Each Frame' code from trigger_image
                    if image_test.status == STARTED and not test_started:
                        win.callOnFlip(dev.activate_line, bitmask=test_start_code)
                        win.callOnFlip(eyetracker.sendMessage, test_start_code)
                        test_started = True
                    
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        trial_image.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in trial_image.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "trial_image" ---
                for thisComponent in trial_image.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for trial_image
                trial_image.tStop = globalClock.getTime(format='float')
                trial_image.tStopRefresh = tThisFlipGlobal
                thisExp.addData('trial_image.stopped', trial_image.tStop)
                # Run 'End Routine' code from setup_test_image
                thisExp.addData('trial_type', thisImageType)  # adding Trial Type to .csv file (1 = match, 0 = non-match)
                thisExp.addData('image_test_fn', image_test_fn)  # adding Test Image Name to .csv file
                thisExp.addData('correct_response', thisCorrectResp)  # adding Correct Response to .csv file
                
                thisImageNumber += 1  # accrue the image number by 1
                
                # check responses
                if key_resp.keys in ['', [], None]:  # No response was made
                    key_resp.keys = None
                practice_sequence.addData('key_resp.keys',key_resp.keys)
                if key_resp.keys != None:  # we had a response
                    practice_sequence.addData('key_resp.rt', key_resp.rt)
                    practice_sequence.addData('key_resp.duration', key_resp.duration)
                # the Routine "trial_image" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "response_feedback" ---
                # create an object to store info about Routine response_feedback
                response_feedback = data.Routine(
                    name='response_feedback',
                    components=[text_feedback],
                )
                response_feedback.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from set_feedback_text
                if not key_resp.keys:
                    feedback_text = 'Respond Faster'
                    feedback_text_color = [-1, -1, -1]
                
                # roll thisImageNumber back 1 since it got += 1 in the last routine
                elif key_resp.keys[0] == thisCorrectResp:  # check the first key pressed
                    feedback_text = 'Correct'
                    feedback_text_color = [-1, 1, -1]
                
                else:
                    feedback_text = 'Incorrect'
                    feedback_text_color = [1, -1, -1]
                
                text_feedback.setColor(feedback_text_color, colorSpace='rgb')
                text_feedback.setText(feedback_text)
                # store start times for response_feedback
                response_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                response_feedback.tStart = globalClock.getTime(format='float')
                response_feedback.status = STARTED
                response_feedback.maxDuration = None
                # keep track of which components have finished
                response_feedbackComponents = response_feedback.components
                for thisComponent in response_feedback.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "response_feedback" ---
                # if trial has changed, end Routine now
                if isinstance(practice_sequence, data.TrialHandler2) and thisPractice_sequence.thisN != practice_sequence.thisTrial.thisN:
                    continueRoutine = False
                response_feedback.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 2.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text_feedback* updates
                    
                    # if text_feedback is starting this frame...
                    if text_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_feedback.frameNStart = frameN  # exact frame index
                        text_feedback.tStart = t  # local t and not account for scr refresh
                        text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_feedback, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        text_feedback.status = STARTED
                        text_feedback.setAutoDraw(True)
                    
                    # if text_feedback is active this frame...
                    if text_feedback.status == STARTED:
                        # update params
                        pass
                    
                    # if text_feedback is stopping this frame...
                    if text_feedback.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_feedback.tStartRefresh + 2.0-frameTolerance:
                            # keep track of stop time/frame for later
                            text_feedback.tStop = t  # not accounting for scr refresh
                            text_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                            text_feedback.frameNStop = frameN  # exact frame index
                            # update status
                            text_feedback.status = FINISHED
                            text_feedback.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        response_feedback.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in response_feedback.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "response_feedback" ---
                for thisComponent in response_feedback.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for response_feedback
                response_feedback.tStop = globalClock.getTime(format='float')
                response_feedback.tStopRefresh = tThisFlipGlobal
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if response_feedback.maxDurationReached:
                    routineTimer.addTime(-response_feedback.maxDuration)
                elif response_feedback.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-2.000000)
                thisExp.nextEntry()
                
            # completed n_images_per_trial repeats of 'practice_sequence'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if practice_sequence.trialList in ([], [None], None):
                params = []
            else:
                params = practice_sequence.trialList[0].keys()
            # save data for this loop
            practice_sequence.saveAsText(filename + 'practice_sequence.csv', delim=',',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            thisExp.nextEntry()
            
        # completed n_trials_practice repeats of 'practice_trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if practice_trials.trialList in ([], [None], None):
            params = []
        else:
            params = practice_trials.trialList[0].keys()
        # save data for this loop
        practice_trials.saveAsText(filename + 'practice_trials.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "practice_checkpoint" ---
        # create an object to store info about Routine practice_checkpoint
        practice_checkpoint = data.Routine(
            name='practice_checkpoint',
            components=[text_checkpoint, key_checkpoint, read_checkpoint],
        )
        practice_checkpoint.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_checkpoint
        key_checkpoint.keys = []
        key_checkpoint.rt = []
        _key_checkpoint_allKeys = []
        read_checkpoint.setSound('resource/instruct_checkpoint.wav', hamming=True)
        read_checkpoint.setVolume(1.0, log=False)
        read_checkpoint.seek(0)
        # Run 'Begin Routine' code from code_checkpoint
        # End of practice block
        dev.activate_line(bitmask=block_end_code)
        eyetracker.sendMessage(block_end_code)
        # no need to wait 500ms as this routine waits for experimenter key press
        
        # store start times for practice_checkpoint
        practice_checkpoint.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_checkpoint.tStart = globalClock.getTime(format='float')
        practice_checkpoint.status = STARTED
        practice_checkpoint.maxDuration = None
        # keep track of which components have finished
        practice_checkpointComponents = practice_checkpoint.components
        for thisComponent in practice_checkpoint.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_checkpoint" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        practice_checkpoint.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_checkpoint* updates
            
            # if text_checkpoint is starting this frame...
            if text_checkpoint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_checkpoint.frameNStart = frameN  # exact frame index
                text_checkpoint.tStart = t  # local t and not account for scr refresh
                text_checkpoint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_checkpoint, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_checkpoint.status = STARTED
                text_checkpoint.setAutoDraw(True)
            
            # if text_checkpoint is active this frame...
            if text_checkpoint.status == STARTED:
                # update params
                pass
            
            # *key_checkpoint* updates
            waitOnFlip = False
            
            # if key_checkpoint is starting this frame...
            if key_checkpoint.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_checkpoint.frameNStart = frameN  # exact frame index
                key_checkpoint.tStart = t  # local t and not account for scr refresh
                key_checkpoint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_checkpoint, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_checkpoint.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_checkpoint.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_checkpoint.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_checkpoint.status == STARTED and not waitOnFlip:
                theseKeys = key_checkpoint.getKeys(keyList=['r', 'o'], ignoreKeys=["escape"], waitRelease=True)
                _key_checkpoint_allKeys.extend(theseKeys)
                if len(_key_checkpoint_allKeys):
                    key_checkpoint.keys = _key_checkpoint_allKeys[-1].name  # just the last key pressed
                    key_checkpoint.rt = _key_checkpoint_allKeys[-1].rt
                    key_checkpoint.duration = _key_checkpoint_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *read_checkpoint* updates
            
            # if read_checkpoint is starting this frame...
            if read_checkpoint.status == NOT_STARTED and tThisFlip >= 0.4-frameTolerance:
                # keep track of start time/frame for later
                read_checkpoint.frameNStart = frameN  # exact frame index
                read_checkpoint.tStart = t  # local t and not account for scr refresh
                read_checkpoint.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                read_checkpoint.status = STARTED
                read_checkpoint.play(when=win)  # sync with win flip
            
            # if read_checkpoint is stopping this frame...
            if read_checkpoint.status == STARTED:
                if bool(False) or read_checkpoint.isFinished:
                    # keep track of stop time/frame for later
                    read_checkpoint.tStop = t  # not accounting for scr refresh
                    read_checkpoint.tStopRefresh = tThisFlipGlobal  # on global time
                    read_checkpoint.frameNStop = frameN  # exact frame index
                    # update status
                    read_checkpoint.status = FINISHED
                    read_checkpoint.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[read_checkpoint]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practice_checkpoint.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_checkpoint.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_checkpoint" ---
        for thisComponent in practice_checkpoint.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_checkpoint
        practice_checkpoint.tStop = globalClock.getTime(format='float')
        practice_checkpoint.tStopRefresh = tThisFlipGlobal
        read_checkpoint.pause()  # ensure sound has stopped at end of Routine
        # Run 'End Routine' code from code_checkpoint
        if key_checkpoint.keys == 'o':  # proceed to main experiment
            practice_loop.finished = True
        
        # the Routine "practice_checkpoint" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 99.0 repeats of 'practice_loop'
    
    
    # --- Prepare to start Routine "instruct_begin" ---
    # create an object to store info about Routine instruct_begin
    instruct_begin = data.Routine(
        name='instruct_begin',
        components=[text_begin, key_begin, read_begin],
    )
    instruct_begin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_begin
    key_begin.keys = []
    key_begin.rt = []
    _key_begin_allKeys = []
    read_begin.setSound('resource/instruct_begin.wav', hamming=True)
    read_begin.setVolume(1.0, log=False)
    read_begin.seek(0)
    # store start times for instruct_begin
    instruct_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_begin.tStart = globalClock.getTime(format='float')
    instruct_begin.status = STARTED
    instruct_begin.maxDuration = None
    # keep track of which components have finished
    instruct_beginComponents = instruct_begin.components
    for thisComponent in instruct_begin.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_begin" ---
    instruct_begin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_begin* updates
        
        # if text_begin is starting this frame...
        if text_begin.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_begin.frameNStart = frameN  # exact frame index
            text_begin.tStart = t  # local t and not account for scr refresh
            text_begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_begin, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_begin.status = STARTED
            text_begin.setAutoDraw(True)
        
        # if text_begin is active this frame...
        if text_begin.status == STARTED:
            # update params
            pass
        
        # *key_begin* updates
        waitOnFlip = False
        
        # if key_begin is starting this frame...
        if key_begin.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_begin.frameNStart = frameN  # exact frame index
            key_begin.tStart = t  # local t and not account for scr refresh
            key_begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_begin, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_begin.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_begin.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_begin.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_begin.status == STARTED and not waitOnFlip:
            theseKeys = key_begin.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
            _key_begin_allKeys.extend(theseKeys)
            if len(_key_begin_allKeys):
                key_begin.keys = _key_begin_allKeys[-1].name  # just the last key pressed
                key_begin.rt = _key_begin_allKeys[-1].rt
                key_begin.duration = _key_begin_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_begin* updates
        
        # if read_begin is starting this frame...
        if read_begin.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_begin.frameNStart = frameN  # exact frame index
            read_begin.tStart = t  # local t and not account for scr refresh
            read_begin.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_begin.status = STARTED
            read_begin.play(when=win)  # sync with win flip
        
        # if read_begin is stopping this frame...
        if read_begin.status == STARTED:
            if bool(False) or read_begin.isFinished:
                # keep track of stop time/frame for later
                read_begin.tStop = t  # not accounting for scr refresh
                read_begin.tStopRefresh = tThisFlipGlobal  # on global time
                read_begin.frameNStop = frameN  # exact frame index
                # update status
                read_begin.status = FINISHED
                read_begin.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_begin]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_begin.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_begin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_begin" ---
    for thisComponent in instruct_begin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_begin
    instruct_begin.tStop = globalClock.getTime(format='float')
    instruct_begin.tStopRefresh = tThisFlipGlobal
    read_begin.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=n_blocks, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "block_setup" ---
        # create an object to store info about Routine block_setup
        block_setup = data.Routine(
            name='block_setup',
            components=[],
        )
        block_setup.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from setup_block
        # Set up the lists for a block
        image_fn_list_this_block = image_fn_list_blocks[blocks.thisRepN]
        imageType_list_this_block = imageType_list_blocks[blocks.thisRepN]
        
        # Start a block of trials
        dev.activate_line(bitmask=block_start_code)
        eyetracker.sendMessage(block_start_code)
        core.wait(0.5)  # wait 500ms before trial triggers
        
        # store start times for block_setup
        block_setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_setup.tStart = globalClock.getTime(format='float')
        block_setup.status = STARTED
        block_setup.maxDuration = None
        # keep track of which components have finished
        block_setupComponents = block_setup.components
        for thisComponent in block_setup.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_setup" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        block_setup.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_setup.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_setup.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_setup" ---
        for thisComponent in block_setup.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_setup
        block_setup.tStop = globalClock.getTime(format='float')
        block_setup.tStopRefresh = tThisFlipGlobal
        # the Routine "block_setup" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=n_trials_per_block, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "trial_setup" ---
            # create an object to store info about Routine trial_setup
            trial_setup = data.Routine(
                name='trial_setup',
                components=[],
            )
            trial_setup.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from setup_trial
            # Set up the content for a trial
            image_fn = image_fn_list_this_block[trials.thisRepN]
            imageType = imageType_list_this_block[trials.thisRepN]
            correct_resps = ['1' if itype == 1 else '2' for itype in imageType]
            
            # store start times for trial_setup
            trial_setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_setup.tStart = globalClock.getTime(format='float')
            trial_setup.status = STARTED
            thisExp.addData('trial_setup.started', trial_setup.tStart)
            trial_setup.maxDuration = None
            # keep track of which components have finished
            trial_setupComponents = trial_setup.components
            for thisComponent in trial_setup.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_setup" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            trial_setup.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial_setup.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_setup.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_setup" ---
            for thisComponent in trial_setup.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_setup
            trial_setup.tStop = globalClock.getTime(format='float')
            trial_setup.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_setup.stopped', trial_setup.tStop)
            # Run 'End Routine' code from setup_trial
            thisExp.addData('target_image', image_fn[1])  # adding Target Image Name to .csv file
            thisExp.addData('distractor_image', image_fn[0])  # adding Distractor Image Name to .csv file
            
            # the Routine "trial_setup" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial_target" ---
            # create an object to store info about Routine trial_target
            trial_target = data.Routine(
                name='trial_target',
                components=[green_square_part1, green_square_part2, image_target, key_resp_catch_delay, text_fixation_first],
            )
            trial_target.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_target.setImage(image_fn[1])
            # Run 'Begin Routine' code from adjust_image_target_size
            box_size = 0.395
            scale_to_size(image_target, box_size)
            
            # create starting attributes for key_resp_catch_delay
            key_resp_catch_delay.keys = []
            key_resp_catch_delay.rt = []
            _key_resp_catch_delay_allKeys = []
            # Run 'Begin Routine' code from initiate_trial_sequence
            # Set the first inter-stimulus interval (ISI) of fixation
            isi = rng.uniform(1.1, 1.4)
            
            # Begin a fresh counter for the number of images
            thisImageNumber = 0
            
            # Run 'Begin Routine' code from trigger_target
            target_started = False
            
            # store start times for trial_target
            trial_target.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_target.tStart = globalClock.getTime(format='float')
            trial_target.status = STARTED
            thisExp.addData('trial_target.started', trial_target.tStart)
            trial_target.maxDuration = None
            # keep track of which components have finished
            trial_targetComponents = trial_target.components
            for thisComponent in trial_target.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_target" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            trial_target.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *green_square_part1* updates
                
                # if green_square_part1 is starting this frame...
                if green_square_part1.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                    # keep track of start time/frame for later
                    green_square_part1.frameNStart = frameN  # exact frame index
                    green_square_part1.tStart = t  # local t and not account for scr refresh
                    green_square_part1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(green_square_part1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_square_part1.started')
                    # update status
                    green_square_part1.status = STARTED
                    green_square_part1.setAutoDraw(True)
                
                # if green_square_part1 is active this frame...
                if green_square_part1.status == STARTED:
                    # update params
                    pass
                
                # if green_square_part1 is stopping this frame...
                if green_square_part1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > green_square_part1.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        green_square_part1.tStop = t  # not accounting for scr refresh
                        green_square_part1.tStopRefresh = tThisFlipGlobal  # on global time
                        green_square_part1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'green_square_part1.stopped')
                        # update status
                        green_square_part1.status = FINISHED
                        green_square_part1.setAutoDraw(False)
                
                # *green_square_part2* updates
                
                # if green_square_part2 is starting this frame...
                if green_square_part2.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                    # keep track of start time/frame for later
                    green_square_part2.frameNStart = frameN  # exact frame index
                    green_square_part2.tStart = t  # local t and not account for scr refresh
                    green_square_part2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(green_square_part2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_square_part2.started')
                    # update status
                    green_square_part2.status = STARTED
                    green_square_part2.setAutoDraw(True)
                
                # if green_square_part2 is active this frame...
                if green_square_part2.status == STARTED:
                    # update params
                    pass
                
                # if green_square_part2 is stopping this frame...
                if green_square_part2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > green_square_part2.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        green_square_part2.tStop = t  # not accounting for scr refresh
                        green_square_part2.tStopRefresh = tThisFlipGlobal  # on global time
                        green_square_part2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'green_square_part2.stopped')
                        # update status
                        green_square_part2.status = FINISHED
                        green_square_part2.setAutoDraw(False)
                
                # *image_target* updates
                
                # if image_target is starting this frame...
                if image_target.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                    # keep track of start time/frame for later
                    image_target.frameNStart = frameN  # exact frame index
                    image_target.tStart = t  # local t and not account for scr refresh
                    image_target.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_target, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_target.started')
                    # update status
                    image_target.status = STARTED
                    image_target.setAutoDraw(True)
                
                # if image_target is active this frame...
                if image_target.status == STARTED:
                    # update params
                    pass
                
                # if image_target is stopping this frame...
                if image_target.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_target.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_target.tStop = t  # not accounting for scr refresh
                        image_target.tStopRefresh = tThisFlipGlobal  # on global time
                        image_target.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_target.stopped')
                        # update status
                        image_target.status = FINISHED
                        image_target.setAutoDraw(False)
                
                # *key_resp_catch_delay* updates
                waitOnFlip = False
                
                # if key_resp_catch_delay is starting this frame...
                if key_resp_catch_delay.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_catch_delay.frameNStart = frameN  # exact frame index
                    key_resp_catch_delay.tStart = t  # local t and not account for scr refresh
                    key_resp_catch_delay.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_catch_delay, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_catch_delay.started')
                    # update status
                    key_resp_catch_delay.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_catch_delay.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_catch_delay.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_catch_delay is stopping this frame...
                if key_resp_catch_delay.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_catch_delay.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_catch_delay.tStop = t  # not accounting for scr refresh
                        key_resp_catch_delay.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_catch_delay.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_catch_delay.stopped')
                        # update status
                        key_resp_catch_delay.status = FINISHED
                        key_resp_catch_delay.status = FINISHED
                if key_resp_catch_delay.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_catch_delay.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_catch_delay_allKeys.extend(theseKeys)
                    if len(_key_resp_catch_delay_allKeys):
                        key_resp_catch_delay.keys = [key.name for key in _key_resp_catch_delay_allKeys]  # storing all keys
                        key_resp_catch_delay.rt = [key.rt for key in _key_resp_catch_delay_allKeys]
                        key_resp_catch_delay.duration = [key.duration for key in _key_resp_catch_delay_allKeys]
                
                # *text_fixation_first* updates
                
                # if text_fixation_first is starting this frame...
                if text_fixation_first.status == NOT_STARTED and image_target.status == FINISHED:
                    # keep track of start time/frame for later
                    text_fixation_first.frameNStart = frameN  # exact frame index
                    text_fixation_first.tStart = t  # local t and not account for scr refresh
                    text_fixation_first.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_fixation_first, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_fixation_first.started')
                    # update status
                    text_fixation_first.status = STARTED
                    text_fixation_first.setAutoDraw(True)
                
                # if text_fixation_first is active this frame...
                if text_fixation_first.status == STARTED:
                    # update params
                    pass
                
                # if text_fixation_first is stopping this frame...
                if text_fixation_first.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_fixation_first.tStartRefresh + isi-frameTolerance:
                        # keep track of stop time/frame for later
                        text_fixation_first.tStop = t  # not accounting for scr refresh
                        text_fixation_first.tStopRefresh = tThisFlipGlobal  # on global time
                        text_fixation_first.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_fixation_first.stopped')
                        # update status
                        text_fixation_first.status = FINISHED
                        text_fixation_first.setAutoDraw(False)
                # Run 'Each Frame' code from trigger_target
                if image_target.status == STARTED and not target_started:
                    win.callOnFlip(dev.activate_line, bitmask=target_start_code)
                    win.callOnFlip(eyetracker.sendMessage, target_start_code)
                    target_started = True
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial_target.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_target.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_target" ---
            for thisComponent in trial_target.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_target
            trial_target.tStop = globalClock.getTime(format='float')
            trial_target.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_target.stopped', trial_target.tStop)
            # check responses
            if key_resp_catch_delay.keys in ['', [], None]:  # No response was made
                key_resp_catch_delay.keys = None
            trials.addData('key_resp_catch_delay.keys',key_resp_catch_delay.keys)
            if key_resp_catch_delay.keys != None:  # we had a response
                trials.addData('key_resp_catch_delay.rt', key_resp_catch_delay.rt)
                trials.addData('key_resp_catch_delay.duration', key_resp_catch_delay.duration)
            # the Routine "trial_target" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            sequence = data.TrialHandler2(
                name='sequence',
                nReps=n_images_per_trial, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(sequence)  # add the loop to the experiment
            thisSequence = sequence.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisSequence.rgb)
            if thisSequence != None:
                for paramName in thisSequence:
                    globals()[paramName] = thisSequence[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisSequence in sequence:
                currentLoop = sequence
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisSequence.rgb)
                if thisSequence != None:
                    for paramName in thisSequence:
                        globals()[paramName] = thisSequence[paramName]
                
                # --- Prepare to start Routine "trial_image" ---
                # create an object to store info about Routine trial_image
                trial_image = data.Routine(
                    name='trial_image',
                    components=[white_square, image_test, text_fixation, key_resp],
                )
                trial_image.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from setup_test_image
                # Obtain the current image filename
                thisImageType = imageType[thisImageNumber]
                image_test_fn = image_fn[thisImageType]
                thisCorrectResp = correct_resps[thisImageNumber]
                
                # Set the inter-stimulus interval (ISI) of fixation
                if thisImageNumber < (n_images_per_trial - 1):
                    isi = rng.uniform(1.1, 1.4)
                else:
                    # skip showing a fixation cross after the last image in sequence
                    text_fixation.status = FINISHED
                    # but still collect behavioral responses for another 1.5s
                    isi = 1.5
                
                image_test.setImage(image_test_fn)
                # Run 'Begin Routine' code from adjust_image_test_size
                scale_to_size(image_test, box_size)
                
                # create starting attributes for key_resp
                key_resp.keys = []
                key_resp.rt = []
                _key_resp_allKeys = []
                # Run 'Begin Routine' code from trigger_image
                test_started = False
                
                # store start times for trial_image
                trial_image.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                trial_image.tStart = globalClock.getTime(format='float')
                trial_image.status = STARTED
                thisExp.addData('trial_image.started', trial_image.tStart)
                trial_image.maxDuration = None
                # keep track of which components have finished
                trial_imageComponents = trial_image.components
                for thisComponent in trial_image.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "trial_image" ---
                # if trial has changed, end Routine now
                if isinstance(sequence, data.TrialHandler2) and thisSequence.thisN != sequence.thisTrial.thisN:
                    continueRoutine = False
                trial_image.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *white_square* updates
                    
                    # if white_square is starting this frame...
                    if white_square.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        white_square.frameNStart = frameN  # exact frame index
                        white_square.tStart = t  # local t and not account for scr refresh
                        white_square.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(white_square, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'white_square.started')
                        # update status
                        white_square.status = STARTED
                        white_square.setAutoDraw(True)
                    
                    # if white_square is active this frame...
                    if white_square.status == STARTED:
                        # update params
                        pass
                    
                    # if white_square is stopping this frame...
                    if white_square.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > white_square.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            white_square.tStop = t  # not accounting for scr refresh
                            white_square.tStopRefresh = tThisFlipGlobal  # on global time
                            white_square.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'white_square.stopped')
                            # update status
                            white_square.status = FINISHED
                            white_square.setAutoDraw(False)
                    
                    # *image_test* updates
                    
                    # if image_test is starting this frame...
                    if image_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_test.frameNStart = frameN  # exact frame index
                        image_test.tStart = t  # local t and not account for scr refresh
                        image_test.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_test, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_test.started')
                        # update status
                        image_test.status = STARTED
                        image_test.setAutoDraw(True)
                    
                    # if image_test is active this frame...
                    if image_test.status == STARTED:
                        # update params
                        pass
                    
                    # if image_test is stopping this frame...
                    if image_test.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_test.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            image_test.tStop = t  # not accounting for scr refresh
                            image_test.tStopRefresh = tThisFlipGlobal  # on global time
                            image_test.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_test.stopped')
                            # update status
                            image_test.status = FINISHED
                            image_test.setAutoDraw(False)
                    
                    # *text_fixation* updates
                    
                    # if text_fixation is starting this frame...
                    if text_fixation.status == NOT_STARTED and image_test.status == FINISHED:
                        # keep track of start time/frame for later
                        text_fixation.frameNStart = frameN  # exact frame index
                        text_fixation.tStart = t  # local t and not account for scr refresh
                        text_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_fixation, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_fixation.started')
                        # update status
                        text_fixation.status = STARTED
                        text_fixation.setAutoDraw(True)
                    
                    # if text_fixation is active this frame...
                    if text_fixation.status == STARTED:
                        # update params
                        pass
                    
                    # if text_fixation is stopping this frame...
                    if text_fixation.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_fixation.tStartRefresh + isi-frameTolerance:
                            # keep track of stop time/frame for later
                            text_fixation.tStop = t  # not accounting for scr refresh
                            text_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                            text_fixation.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_fixation.stopped')
                            # update status
                            text_fixation.status = FINISHED
                            text_fixation.setAutoDraw(False)
                    
                    # *key_resp* updates
                    waitOnFlip = False
                    
                    # if key_resp is starting this frame...
                    if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp.frameNStart = frameN  # exact frame index
                        key_resp.tStart = t  # local t and not account for scr refresh
                        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.started')
                        # update status
                        key_resp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_resp is stopping this frame...
                    if key_resp.status == STARTED:
                        # is it time to stop? (based on local clock)
                        if tThisFlip > isi + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp.tStop = t  # not accounting for scr refresh
                            key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_resp.stopped')
                            # update status
                            key_resp.status = FINISHED
                            key_resp.status = FINISHED
                    if key_resp.status == STARTED and not waitOnFlip:
                        theseKeys = key_resp.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_allKeys.extend(theseKeys)
                        if len(_key_resp_allKeys):
                            key_resp.keys = [key.name for key in _key_resp_allKeys]  # storing all keys
                            key_resp.rt = [key.rt for key in _key_resp_allKeys]
                            key_resp.duration = [key.duration for key in _key_resp_allKeys]
                    # Run 'Each Frame' code from trigger_image
                    if image_test.status == STARTED and not test_started:
                        win.callOnFlip(dev.activate_line, bitmask=test_start_code)
                        win.callOnFlip(eyetracker.sendMessage, test_start_code)
                        test_started = True
                    
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        trial_image.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in trial_image.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "trial_image" ---
                for thisComponent in trial_image.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for trial_image
                trial_image.tStop = globalClock.getTime(format='float')
                trial_image.tStopRefresh = tThisFlipGlobal
                thisExp.addData('trial_image.stopped', trial_image.tStop)
                # Run 'End Routine' code from setup_test_image
                thisExp.addData('trial_type', thisImageType)  # adding Trial Type to .csv file (1 = match, 0 = non-match)
                thisExp.addData('image_test_fn', image_test_fn)  # adding Test Image Name to .csv file
                thisExp.addData('correct_response', thisCorrectResp)  # adding Correct Response to .csv file
                
                thisImageNumber += 1  # accrue the image number by 1
                
                # check responses
                if key_resp.keys in ['', [], None]:  # No response was made
                    key_resp.keys = None
                sequence.addData('key_resp.keys',key_resp.keys)
                if key_resp.keys != None:  # we had a response
                    sequence.addData('key_resp.rt', key_resp.rt)
                    sequence.addData('key_resp.duration', key_resp.duration)
                # the Routine "trial_image" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed n_images_per_trial repeats of 'sequence'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if sequence.trialList in ([], [None], None):
                params = []
            else:
                params = sequence.trialList[0].keys()
            # save data for this loop
            sequence.saveAsText(filename + 'sequence.csv', delim=',',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            thisExp.nextEntry()
            
        # completed n_trials_per_block repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if trials.trialList in ([], [None], None):
            params = []
        else:
            params = trials.trialList[0].keys()
        # save data for this loop
        trials.saveAsText(filename + 'trials.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "block_cleanup" ---
        # create an object to store info about Routine block_cleanup
        block_cleanup = data.Routine(
            name='block_cleanup',
            components=[],
        )
        block_cleanup.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from cleanup_block
        # End of block
        dev.activate_line(bitmask=block_end_code)
        eyetracker.sendMessage(block_end_code)
        # no need to wait 500ms as we will calibrate again or we have thank you routine
        
        if blocks.thisRepN == (n_blocks - 1):  # last block of trials
            whether_rest = 0  # skip the rest_loop
        else:
            whether_rest = 1  # enter the rest_loop
        
        # store start times for block_cleanup
        block_cleanup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_cleanup.tStart = globalClock.getTime(format='float')
        block_cleanup.status = STARTED
        block_cleanup.maxDuration = None
        # keep track of which components have finished
        block_cleanupComponents = block_cleanup.components
        for thisComponent in block_cleanup.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_cleanup" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        block_cleanup.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_cleanup.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_cleanup.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_cleanup" ---
        for thisComponent in block_cleanup.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_cleanup
        block_cleanup.tStop = globalClock.getTime(format='float')
        block_cleanup.tStopRefresh = tThisFlipGlobal
        # the Routine "block_cleanup" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        rest_loop = data.TrialHandler2(
            name='rest_loop',
            nReps=whether_rest, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(rest_loop)  # add the loop to the experiment
        thisRest_loop = rest_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRest_loop.rgb)
        if thisRest_loop != None:
            for paramName in thisRest_loop:
                globals()[paramName] = thisRest_loop[paramName]
        
        for thisRest_loop in rest_loop:
            currentLoop = rest_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisRest_loop.rgb)
            if thisRest_loop != None:
                for paramName in thisRest_loop:
                    globals()[paramName] = thisRest_loop[paramName]
            
            # --- Prepare to start Routine "instruct_rest" ---
            # create an object to store info about Routine instruct_rest
            instruct_rest = data.Routine(
                name='instruct_rest',
                components=[text_rest, key_rest, read_rest],
            )
            instruct_rest.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_rest
            key_rest.keys = []
            key_rest.rt = []
            _key_rest_allKeys = []
            read_rest.setSound('resource/instruct_rest.wav', hamming=True)
            read_rest.setVolume(1.0, log=False)
            read_rest.seek(0)
            # store start times for instruct_rest
            instruct_rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instruct_rest.tStart = globalClock.getTime(format='float')
            instruct_rest.status = STARTED
            instruct_rest.maxDuration = None
            # keep track of which components have finished
            instruct_restComponents = instruct_rest.components
            for thisComponent in instruct_rest.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instruct_rest" ---
            # if trial has changed, end Routine now
            if isinstance(rest_loop, data.TrialHandler2) and thisRest_loop.thisN != rest_loop.thisTrial.thisN:
                continueRoutine = False
            instruct_rest.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_rest* updates
                
                # if text_rest is starting this frame...
                if text_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_rest.frameNStart = frameN  # exact frame index
                    text_rest.tStart = t  # local t and not account for scr refresh
                    text_rest.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_rest, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_rest.status = STARTED
                    text_rest.setAutoDraw(True)
                
                # if text_rest is active this frame...
                if text_rest.status == STARTED:
                    # update params
                    pass
                
                # *key_rest* updates
                waitOnFlip = False
                
                # if key_rest is starting this frame...
                if key_rest.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                    # keep track of start time/frame for later
                    key_rest.frameNStart = frameN  # exact frame index
                    key_rest.tStart = t  # local t and not account for scr refresh
                    key_rest.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_rest, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_rest.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_rest.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_rest.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_rest.status == STARTED and not waitOnFlip:
                    theseKeys = key_rest.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
                    _key_rest_allKeys.extend(theseKeys)
                    if len(_key_rest_allKeys):
                        key_rest.keys = _key_rest_allKeys[-1].name  # just the last key pressed
                        key_rest.rt = _key_rest_allKeys[-1].rt
                        key_rest.duration = _key_rest_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *read_rest* updates
                
                # if read_rest is starting this frame...
                if read_rest.status == NOT_STARTED and tThisFlip >= 0.4-frameTolerance:
                    # keep track of start time/frame for later
                    read_rest.frameNStart = frameN  # exact frame index
                    read_rest.tStart = t  # local t and not account for scr refresh
                    read_rest.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    read_rest.status = STARTED
                    read_rest.play(when=win)  # sync with win flip
                
                # if read_rest is stopping this frame...
                if read_rest.status == STARTED:
                    if bool(False) or read_rest.isFinished:
                        # keep track of stop time/frame for later
                        read_rest.tStop = t  # not accounting for scr refresh
                        read_rest.tStopRefresh = tThisFlipGlobal  # on global time
                        read_rest.frameNStop = frameN  # exact frame index
                        # update status
                        read_rest.status = FINISHED
                        read_rest.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[read_rest]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instruct_rest.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instruct_rest.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instruct_rest" ---
            for thisComponent in instruct_rest.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instruct_rest
            instruct_rest.tStop = globalClock.getTime(format='float')
            instruct_rest.tStopRefresh = tThisFlipGlobal
            read_rest.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instruct_rest" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "_et_instruct" ---
            # create an object to store info about Routine _et_instruct
            _et_instruct = data.Routine(
                name='_et_instruct',
                components=[text_et, key_et, read_et],
            )
            _et_instruct.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_et
            key_et.keys = []
            key_et.rt = []
            _key_et_allKeys = []
            read_et.setSound('resource/eyetrack_calibrate_instruct.wav', hamming=True)
            read_et.setVolume(1.0, log=False)
            read_et.seek(0)
            # store start times for _et_instruct
            _et_instruct.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            _et_instruct.tStart = globalClock.getTime(format='float')
            _et_instruct.status = STARTED
            _et_instruct.maxDuration = None
            # keep track of which components have finished
            _et_instructComponents = _et_instruct.components
            for thisComponent in _et_instruct.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "_et_instruct" ---
            # if trial has changed, end Routine now
            if isinstance(rest_loop, data.TrialHandler2) and thisRest_loop.thisN != rest_loop.thisTrial.thisN:
                continueRoutine = False
            _et_instruct.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_et* updates
                
                # if text_et is starting this frame...
                if text_et.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_et.frameNStart = frameN  # exact frame index
                    text_et.tStart = t  # local t and not account for scr refresh
                    text_et.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_et, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_et.status = STARTED
                    text_et.setAutoDraw(True)
                
                # if text_et is active this frame...
                if text_et.status == STARTED:
                    # update params
                    pass
                
                # *key_et* updates
                waitOnFlip = False
                
                # if key_et is starting this frame...
                if key_et.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                    # keep track of start time/frame for later
                    key_et.frameNStart = frameN  # exact frame index
                    key_et.tStart = t  # local t and not account for scr refresh
                    key_et.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_et, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_et.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_et.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_et.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_et.status == STARTED and not waitOnFlip:
                    theseKeys = key_et.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
                    _key_et_allKeys.extend(theseKeys)
                    if len(_key_et_allKeys):
                        key_et.keys = _key_et_allKeys[-1].name  # just the last key pressed
                        key_et.rt = _key_et_allKeys[-1].rt
                        key_et.duration = _key_et_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *read_et* updates
                
                # if read_et is starting this frame...
                if read_et.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                    # keep track of start time/frame for later
                    read_et.frameNStart = frameN  # exact frame index
                    read_et.tStart = t  # local t and not account for scr refresh
                    read_et.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    read_et.status = STARTED
                    read_et.play(when=win)  # sync with win flip
                
                # if read_et is stopping this frame...
                if read_et.status == STARTED:
                    if bool(False) or read_et.isFinished:
                        # keep track of stop time/frame for later
                        read_et.tStop = t  # not accounting for scr refresh
                        read_et.tStopRefresh = tThisFlipGlobal  # on global time
                        read_et.frameNStop = frameN  # exact frame index
                        # update status
                        read_et.status = FINISHED
                        read_et.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[read_et]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    _et_instruct.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in _et_instruct.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "_et_instruct" ---
            for thisComponent in _et_instruct.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for _et_instruct
            _et_instruct.tStop = globalClock.getTime(format='float')
            _et_instruct.tStopRefresh = tThisFlipGlobal
            read_et.pause()  # ensure sound has stopped at end of Routine
            # the Routine "_et_instruct" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "_et_mask" ---
            # create an object to store info about Routine _et_mask
            _et_mask = data.Routine(
                name='_et_mask',
                components=[text_mask],
            )
            _et_mask.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for _et_mask
            _et_mask.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            _et_mask.tStart = globalClock.getTime(format='float')
            _et_mask.status = STARTED
            _et_mask.maxDuration = None
            # keep track of which components have finished
            _et_maskComponents = _et_mask.components
            for thisComponent in _et_mask.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "_et_mask" ---
            # if trial has changed, end Routine now
            if isinstance(rest_loop, data.TrialHandler2) and thisRest_loop.thisN != rest_loop.thisTrial.thisN:
                continueRoutine = False
            _et_mask.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.05:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_mask* updates
                
                # if text_mask is starting this frame...
                if text_mask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_mask.frameNStart = frameN  # exact frame index
                    text_mask.tStart = t  # local t and not account for scr refresh
                    text_mask.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_mask, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_mask.status = STARTED
                    text_mask.setAutoDraw(True)
                
                # if text_mask is active this frame...
                if text_mask.status == STARTED:
                    # update params
                    pass
                
                # if text_mask is stopping this frame...
                if text_mask.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_mask.tStartRefresh + 0.05-frameTolerance:
                        # keep track of stop time/frame for later
                        text_mask.tStop = t  # not accounting for scr refresh
                        text_mask.tStopRefresh = tThisFlipGlobal  # on global time
                        text_mask.frameNStop = frameN  # exact frame index
                        # update status
                        text_mask.status = FINISHED
                        text_mask.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    _et_mask.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in _et_mask.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "_et_mask" ---
            for thisComponent in _et_mask.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for _et_mask
            _et_mask.tStop = globalClock.getTime(format='float')
            _et_mask.tStopRefresh = tThisFlipGlobal
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if _et_mask.maxDurationReached:
                routineTimer.addTime(-_et_mask.maxDuration)
            elif _et_mask.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.050000)
            # define target for _et_cal
            _et_calTarget = visual.TargetStim(win, 
                name='_et_calTarget',
                radius=0.015, fillColor='white', borderColor='green', lineWidth=2.0,
                innerRadius=0.005, innerFillColor='black', innerBorderColor='black', innerLineWidth=2.0,
                colorSpace='rgb', units=None
            )
            # define parameters for _et_cal
            _et_cal = hardware.eyetracker.EyetrackerCalibration(win, 
                eyetracker, _et_calTarget,
                units=None, colorSpace='rgb',
                progressMode='time', targetDur=1.5, expandScale=1.5,
                targetLayout='NINE_POINTS', randomisePos=True, textColor='white',
                movementAnimation=True, targetDelay=1.0
            )
            # run calibration
            _et_cal.run()
            # clear any keypresses from during _et_cal so they don't interfere with the experiment
            defaultKeyboard.clearEvents()
            # the Routine "_et_cal" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instruct_continue" ---
            # create an object to store info about Routine instruct_continue
            instruct_continue = data.Routine(
                name='instruct_continue',
                components=[text_continue, key_continue, read_continue],
            )
            instruct_continue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_continue
            key_continue.keys = []
            key_continue.rt = []
            _key_continue_allKeys = []
            read_continue.setSound('resource/instruct_continue.wav', hamming=True)
            read_continue.setVolume(1.0, log=False)
            read_continue.seek(0)
            # store start times for instruct_continue
            instruct_continue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instruct_continue.tStart = globalClock.getTime(format='float')
            instruct_continue.status = STARTED
            instruct_continue.maxDuration = None
            # keep track of which components have finished
            instruct_continueComponents = instruct_continue.components
            for thisComponent in instruct_continue.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instruct_continue" ---
            # if trial has changed, end Routine now
            if isinstance(rest_loop, data.TrialHandler2) and thisRest_loop.thisN != rest_loop.thisTrial.thisN:
                continueRoutine = False
            instruct_continue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_continue* updates
                
                # if text_continue is starting this frame...
                if text_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_continue.frameNStart = frameN  # exact frame index
                    text_continue.tStart = t  # local t and not account for scr refresh
                    text_continue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_continue, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_continue.status = STARTED
                    text_continue.setAutoDraw(True)
                
                # if text_continue is active this frame...
                if text_continue.status == STARTED:
                    # update params
                    pass
                
                # *key_continue* updates
                waitOnFlip = False
                
                # if key_continue is starting this frame...
                if key_continue.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                    # keep track of start time/frame for later
                    key_continue.frameNStart = frameN  # exact frame index
                    key_continue.tStart = t  # local t and not account for scr refresh
                    key_continue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_continue, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_continue.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_continue.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_continue.status == STARTED and not waitOnFlip:
                    theseKeys = key_continue.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
                    _key_continue_allKeys.extend(theseKeys)
                    if len(_key_continue_allKeys):
                        key_continue.keys = _key_continue_allKeys[-1].name  # just the last key pressed
                        key_continue.rt = _key_continue_allKeys[-1].rt
                        key_continue.duration = _key_continue_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *read_continue* updates
                
                # if read_continue is starting this frame...
                if read_continue.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                    # keep track of start time/frame for later
                    read_continue.frameNStart = frameN  # exact frame index
                    read_continue.tStart = t  # local t and not account for scr refresh
                    read_continue.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    read_continue.status = STARTED
                    read_continue.play(when=win)  # sync with win flip
                
                # if read_continue is stopping this frame...
                if read_continue.status == STARTED:
                    if bool(False) or read_continue.isFinished:
                        # keep track of stop time/frame for later
                        read_continue.tStop = t  # not accounting for scr refresh
                        read_continue.tStopRefresh = tThisFlipGlobal  # on global time
                        read_continue.frameNStop = frameN  # exact frame index
                        # update status
                        read_continue.status = FINISHED
                        read_continue.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[read_continue]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instruct_continue.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instruct_continue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instruct_continue" ---
            for thisComponent in instruct_continue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instruct_continue
            instruct_continue.tStop = globalClock.getTime(format='float')
            instruct_continue.tStopRefresh = tThisFlipGlobal
            read_continue.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instruct_continue" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed whether_rest repeats of 'rest_loop'
        
    # completed n_blocks repeats of 'blocks'
    
    
    # --- Prepare to start Routine "__end__" ---
    # create an object to store info about Routine __end__
    __end__ = data.Routine(
        name='__end__',
        components=[text_thank_you, read_thank_you],
    )
    __end__.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    read_thank_you.setSound('resource/thank_you.wav', secs=2.7, hamming=True)
    read_thank_you.setVolume(1.0, log=False)
    read_thank_you.seek(0)
    # store start times for __end__
    __end__.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    __end__.tStart = globalClock.getTime(format='float')
    __end__.status = STARTED
    __end__.maxDuration = None
    # keep track of which components have finished
    __end__Components = __end__.components
    for thisComponent in __end__.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__end__" ---
    __end__.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_thank_you* updates
        
        # if text_thank_you is starting this frame...
        if text_thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_thank_you.frameNStart = frameN  # exact frame index
            text_thank_you.tStart = t  # local t and not account for scr refresh
            text_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_thank_you, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_thank_you.status = STARTED
            text_thank_you.setAutoDraw(True)
        
        # if text_thank_you is active this frame...
        if text_thank_you.status == STARTED:
            # update params
            pass
        
        # if text_thank_you is stopping this frame...
        if text_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_thank_you.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                text_thank_you.tStop = t  # not accounting for scr refresh
                text_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                text_thank_you.frameNStop = frameN  # exact frame index
                # update status
                text_thank_you.status = FINISHED
                text_thank_you.setAutoDraw(False)
        
        # *read_thank_you* updates
        
        # if read_thank_you is starting this frame...
        if read_thank_you.status == NOT_STARTED and t >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            read_thank_you.frameNStart = frameN  # exact frame index
            read_thank_you.tStart = t  # local t and not account for scr refresh
            read_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_thank_you.status = STARTED
            read_thank_you.play()  # start the sound (it finishes automatically)
        
        # if read_thank_you is stopping this frame...
        if read_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > read_thank_you.tStartRefresh + 2.7-frameTolerance or read_thank_you.isFinished:
                # keep track of stop time/frame for later
                read_thank_you.tStop = t  # not accounting for scr refresh
                read_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                read_thank_you.frameNStop = frameN  # exact frame index
                # update status
                read_thank_you.status = FINISHED
                read_thank_you.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_thank_you]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            __end__.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __end__.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__end__" ---
    for thisComponent in __end__.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for __end__
    __end__.tStop = globalClock.getTime(format='float')
    __end__.tStopRefresh = tThisFlipGlobal
    read_thank_you.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if __end__.maxDurationReached:
        routineTimer.addTime(-__end__.maxDuration)
    elif __end__.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from eeg
    # Stop EEG recording
    dev.activate_line(bitmask=127)  # trigger 127 will stop EEG
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
