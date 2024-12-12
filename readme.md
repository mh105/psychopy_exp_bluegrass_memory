# Bluegrass working memory task
Last edit: 12/11/2024

## Edit history
- 12/11/2024 by Alex He - fixed the window size to be the correct screen resolution
- 11/22/2024 by Alex He - removed summary csv saving since no trialList used
- 11/21/2024 by Alex He - added voice over instruction to encourage questions on the schematic diagram
- 10/24/2024 by Alex He - added a print message of task ID at the onset of task
- 10/16/2024 by Emily McElroy - fixed a bug with sequence loop not treated as a trial loop
- 10/12/2024 by Alex He - increased logging granularity from warning to debug (maximal level)
- 10/10/2024 by Alex He - added MilliKey response box and finalized voice-over audio
- 09/25/2024 by Alex He - added winHandle.activate() to make sure window is on foreground
- 09/23/2024 by Alex He - upgraded to run on PsychoPy 2024.2.2
- 09/18/2024 by Alex He - improved trial logging and enabled two repeated test sessions
- 09/16/2024 by Alex He - fixed 6 stimuli picture with non-white (255) background
- 09/05/2024 by Alex He - removed git tracking of _lastrun.py file and added retries to pyxid2.get_xid_devices() with timeout
- 08/17/2024 by Alex He - added more print messages during c-pod connection
- 08/13/2024 by Alex He - added reminder that real trials do not have feedback on responses
- 08/12/2024 by Alex He - reverted to python 3.8 as pylink connection to EyeLink does not work correctly on 3.10
- 08/04/2024 by Alex He - generated experiment scripts on python 3.10
- 08/02/2024 by Alex He - upgraded to support PsychoPy 2024.2.1
- 07/26/2024 by Alex He - upgraded to support PsychoPy 2024.2.0
- 07/23/2024 by Emily McElroy - fixed three minor bugs around the maximal number of image file loaded, thisRepN to skip extra calibration, and setting inner most loop isTrial=True
- 07/15/2024 by Alex He - updated task schematic diagram png file
- 06/30/2024 by Alex He - created finalized first draft version

## Description
This is a modified delayed match-to-sample task used for over a decade by Dr. Yang Jiang's group at University of Kentucky. The behavioral paradigm has a simple design where in each sequence subjects are asked to make a match/no-match two-alternative forced choice response against a target. Visual stimuli have been employed in previous studies, and the focus is on memory-related frontal potentials that have been found to be sensitive to aging and Alzheimer's disease (AD) pathology, including mild cognitive impairment (MCI), longitudinal cognitive decline, as well as AD diagnosis.

For a detailed description of this line of research, see the very helpful "Human Studies using Bluegrass Memory Paradigm" PDF file written by Jiang's group in the current repo. Our design here follows the 2017 study that differentiated aMCI from normal cognitively normal older adults:

Li, J., Broster, L. S., Jicha, G. A., Munro, N. B., Schmitt, F. A., Abner, E., ... & Jiang, Y. (2017). A cognitive electrophysiological signature differentiates amnestic mild cognitive impairment from normal aging. Alzheimer's research & therapy, 9, 1-10.

And we decided to use this paradigm because of the more recent report of predictability of transition from normal aging to MCI by the following paper:

Jiang, Y., Li, J., Schmitt, F. A., Jicha, G. A., Munro, N. B., Zhao, X., ... & Abner, E. L. (2021). Memory-related frontal brainwaves predict transition to mild cognitive impairment in healthy older individuals five years before diagnosis. Journal of Alzheimer's Disease, 79(2), 531-541.

We have adopted the design as closely as possible including the number of trials, blocks, and trial timings, in order to replicate the previous findings and to explore the added analytic power with our state-space modeling approach to extract event-related potentials (ERP).

## Outcome measures
- Left frontal memory-related brain potentials that differ between match and non-match (see the PDF file for more details)
