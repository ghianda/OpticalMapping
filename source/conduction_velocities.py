# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

# lista colori disponibili per xlwt in:
# https://github.com/python-excel/xlwt/blob/master/xlwt/Style.py  (riga 300 circa)
# ciao test git

import os
import argparse

# data manipulation
import numpy as np
import scipy.signal as sign
import scipy.stats as stats
from scipy import interpolate

# excel manipulation
import xlwt
from xlwt import Workbook

# plot
import matplotlib
import matplotlib.pyplot as plt

# plot settings
matplotlib.use('Qt5Agg')
plt.rcParams["figure.figsize"] = (15, 5)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class pKeys:
    frame_rate      = 'frame_rate'
    ROI_distance_mm = 'ROI_distance_mm'
    rank            = 'rank'
    exp_duration    = 'exp_duration'

class bKeys:
    freq_stim   = 'freq_stim'
    AP_duration = 'AP_duration'
    start_ms    = 'start_ms'
    stop_ms     = 'stop_ms'



def lag_finder_in_ms(y1, y2, sr, _plot=False):
    n = len(y1)

    corr = sign.correlate(y2, y1, mode='same') / np.sqrt(
        sign.correlate(y1, y1, mode='same')[int(n / 2)] * sign.correlate(y2, y2, mode='same')[int(n / 2)])

    delay_arr = np.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
    delay = delay_arr[np.argmax(corr)]

    if _plot:
        plt.figure()
        plt.plot(delay_arr, corr)
        plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coeff')
        plt.show()
    return delay * 1000


def plot_signals(apex, base, title=''):
    plt.plot(apex , label='apex')
    plt.plot(base , label='base')
    plt.title(title)
    plt.xlabel('ms'), plt.ylabel('Intensity')
    plt.legend(), plt.show()


def extract_info(filename, _print=False):
    # extract general parameters
    splitted_ = filename.split('_')
    param = dict()
    param[pKeys.ROI_distance_mm] = np.float(splitted_[2][7:11])
    param[pKeys.frame_rate] = np.float(splitted_[3][5:])
    param[pKeys.rank] = np.int(splitted_[4][4:])  # rank od 1D median filter (DISPARI)

    # extract frequencies of stimulation
    splitted_hz = filename.split('Hz')

    # def structured data containing burst information
    burst = np.zeros(len(splitted_hz) - 1,
                     dtype=[('freq_stim', np.int),  # Frequency of the stimultation burst
                            ('start_ms', np.int64),
                            # Start of the record portion containing current stimolation frequency
                            ('stop_ms', np.int64),
                            # Stop of the record portion containing current stimolation frequency
                            ('AP_duration', np.float16)  # Average AP duration at the current stim frequencies
                            ])

    # compile burst information
    for i in range(0, len(splitted_hz) - 1):
        burst[i][bKeys.freq_stim] = np.int(splitted_hz[i].split('_')[-1])
        burst[i][bKeys.AP_duration] = 1000 * (1 / burst[i][bKeys.freq_stim])
        burst[i][bKeys.start_ms] = splitted_hz[i].split('_')[-3]
        burst[i][bKeys.stop_ms] = splitted_hz[i].split('_')[-2]

    if _print:
        # Print results
        print("- Frame Rate               : {} fps".format(param[pKeys.frame_rate]))
        print("- Roi distance             : {} mm".format(param[pKeys.ROI_distance_mm]))
        print("- Rank of median filter    : {} ".format(param[pKeys.rank]))
        print("\n- Burst of stimuls selected:")
        for b in burst:
            print()
            print("-- Stimulation frequency: {} Hz".format(b[bKeys.freq_stim]))
            print("-- Ramp extremes        : from {}ms to {}ms".format(b[bKeys.start_ms], b[bKeys.stop_ms]))
            print("-- Average AP duration  : {} ms".format(b[bKeys.AP_duration]))

    return burst, param


def plot_signals_with_peaks(signals, peaks_pos, labels, title, _enum_peaks=False, _plot_intervals=False,
                            frame_rate=None, intervals_of_signal_n=None):
    # signals - list of signals to be plotted
    # peaks_pos - list of array of peaks positions to be highlighted
    # example: signals = [apex, base] and peaks_pos = [peaks_apex, peaks_base]
    #
    # NB - plt.ion() and  plt.pause(0.01) before and after plt.show()
    # allow to enable the "interactive-mode" for matplotlib:
    # when firts plot is showed, execution in the shell do not stop and
    # user can insert values in the command line and go ahead
    # and the execution continues.

    # convert input parameters to list if they aren't already lists
    signals = [signals] if type(signals) is not list else signals
    peaks_pos = [peaks_pos] if type(peaks_pos) is not list else peaks_pos
    labels = [labels] if type(labels) is not list else labels

    min_y = 0  # used to enlarge y axis at the bottom

    if len(signals) == len(peaks_pos):
        fig, ax = plt.subplots()
        plt.ion()  #added for enable "non-block" mode

        # plot signals and peaks
        for (s, pos, lab) in zip(signals, peaks_pos, labels):
            ax.plot(s, label=lab)
            ax.plot(pos, s[pos], "x")
            min_y = min(s) if min(s) < min_y else min_y

            if _enum_peaks is True:
                for (i, p) in enumerate(pos):
                    plt.text(p - 30, s[p] + s[p] / 10, str(i))

        if _plot_intervals and frame_rate is not None:
            # plot period 5durations

            # if not already selected with 'intervals_of_signal_n',
            # search signal with lowest minimum value
            # to use as position reference to plot lines and texts
            if intervals_of_signal_n is None:
                if len(signals) > 1:
                    intervals_of_signal_n = np.argmin(
                        np.array([np.min(s) for s in signals]))
                else:
                    intervals_of_signal_n = 0
            sign = signals[intervals_of_signal_n]  # select only first signal to show periods
            p_pos = peaks_pos[intervals_of_signal_n]  # select peaks of only first signal to show periods
            ap_durations = np.diff(p_pos) / frame_rate

            # plot AP durations as lines and texts
            for (i, ap) in enumerate(ap_durations):
                # line position
                y_info_pos = sign[p_pos[i]] + (2 / 3) * sign[p_pos[i]]
                x_info_min = p_pos[i] + p_pos[i] / 100
                x_info_max = p_pos[i + 1] - p_pos[i - 1] / 100

                # plot segent
                plt.hlines(y=y_info_pos, xmin=x_info_min, xmax=x_info_max)

                # print duration below the line
                plt.text(x=x_info_min,
                         y=y_info_pos + y_info_pos / 10,
                         s="{0} ms".format(int(ap)))

                # update min_y to enlarge plot
                min_y = y_info_pos if y_info_pos < min_y else min_y

        plt.title(title)
        plt.xlabel('ms'), plt.ylabel('Intensity')
        ax.set_ylim(min_y + min_y / 3, None)
        plt.legend()
        plt.show()
        plt.pause(0.01)  # added for enable "non-block" mode

    else:
        print('ERROR - different number of signals and peaks passed to plot')


def prepare_excel_wb(folderpath, exp_filename, burst, param, out_name=None):
    # create workbook
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet = wb.add_sheet('Sheet_1')

    # create filename and filepath
    out_name = ('_').join(exp_filename.split('_')[0:5] + ['results.xls']) if out_name is None else out_name
    out_path = os.path.join(folderpath, out_name)

    # write titles
    sheet.write_merge(0, 0, 0, 10, "TXT FILENAME  : {}".format(exp_filename))
    sheet.write_merge(1, 1, 0, 10, "EXP PATH          : {}".format(folderpath))

    # write "mean" and "std" cells at side of the sheet
    sides = ['python mean', 'python sem', 'excel mean', 'excel sem']
    for (i, s) in enumerate(sides):
        sheet.write(17 + i, 0, s)

    # define names of columns
    fields = ['AP num', 'CV py', 'CV lab', 'CV map', 'TTP', '50%', '90%', 'APA']
    n_col = len(fields)

    # prepare a block of cells for each burst of stimuli
    # each block is separated by 1 empty column (-> block_index * (columns + 1))
    # and start from column B (-> block_index * (columns + 1) + 1)
    for (ib, b) in enumerate(burst):

        # write block title and start/stop in ms
        sheet.write_merge(3, 3, (ib * (n_col + 1)) + 1, ((ib + 1) * (n_col + 1) - 1),
                          "eStim {}Hz:".format(b[bKeys.freq_stim]),
                          xlwt.easyxf("align: horiz center; pattern: pattern solid, fore_colour gray25"))

        sheet.write_merge(4, 4, (ib * (n_col + 1)) + 1, ((ib + 1) * (n_col + 1) - 1),
                          "from {}ms to {}ms".format(b[bKeys.start_ms], b[bKeys.stop_ms]),
                          xlwt.easyxf("align: horiz center"))

        # write measure units
        sheet.write(5, (ib * (n_col + 1)) + 1, " - ", xlwt.easyxf("align: horiz center"))
        sheet.write_merge(5, 5, (ib * (n_col + 1)) + 2, (ib * (n_col + 1)) + 4, "(m/s)", xlwt.easyxf("align: horiz center"))
        sheet.write_merge(5, 5, (ib * (n_col + 1)) + 5, (ib * (n_col + 1)) + 7, "(ms)", xlwt.easyxf("align: horiz center"))
        sheet.write(5, (ib * (n_col + 1)) + 8, "%", xlwt.easyxf("align: horiz center"))

        # write fields names
        for (i_field, fname) in enumerate(fields):
            sheet.write(6, (ib * (n_col + 1)) + i_field + 1, fname, xlwt.easyxf("align: horiz center"))

    # save excel file
    wb.save(out_path)

    return wb, sheet, out_path, n_col


def main(parser):

    # print(bcolors.HEADER + "*** Conduction Velocity Analyzer ***" + bcolors.ENDC) # on windows cmd colors don't work
    print('\n***************************************')
    print('*** Conduction Velocity Analyzer ******')
    print('\n***************************************')

    # parse arguments
    args = parser.parse_args()

    txt_path = args.input_filepath[0]

    # extract filename and directory path
    filename     = os.path.basename(txt_path)
    folderpath   = os.path.dirname(txt_path)

    # extract experiment parameters from filename and print info to console
    burst, param = extract_info(filename, _print=True)

    # create excel file
    wb, sheet, out_path, n_col_sheet_block = prepare_excel_wb(folderpath, filename, burst, param)

    # read OM tracks values
    values = np.loadtxt(txt_path, dtype=np.float, usecols=(0, 1, 2))
    param[pKeys.exp_duration] = values.shape[0] / param[pKeys.frame_rate]
    print("- Duration of record : {} ms".format(param[pKeys.exp_duration]))

    # split timing, roi1 (apex) and roi2 (base) values
    full_ms, full_apex, full_base = values[:, 0], values[:, 1], values[:, 2]

    ''' ===== [ START ANALISIS ] ===== - for each burst '''
    for (i_burst, b) in enumerate(burst):

        print('\n\n********* Analyzing Burst at {}Hz *********'.format(b[bKeys.freq_stim]))

        # extract ramp
        ms = full_ms[b[bKeys.start_ms]:b[bKeys.stop_ms]]
        apex = full_apex[b[bKeys.start_ms]:b[bKeys.stop_ms]]
        base = full_base[b[bKeys.start_ms]:b[bKeys.stop_ms]]
        # plot_signals(apex, base, title="Ramp at {}Hz".format(b[keys.freq_stim]))

        # detrend signals
        apex_flat = sign.detrend(apex, type='linear')
        base_flat = sign.detrend(base, type='linear')

        # plot original and flattened
        # plot_signals(apex, base, title="Original Tracks")
        # plot_signals(apex_flat, base_flat, title="Detrended Tracks")

        # if selected, apply median filter and plot results
        if param[pKeys.rank] is not 0:
            apex_filt = sign.medfilt(apex_flat, kernel_size=param[pKeys.rank])
            base_filt = sign.medfilt(base_flat, kernel_size=param[pKeys.rank])
            # plot_signals(apex_filt, base_filt, title="Filtered Tracks (rank = {})".format(param[keys.rank]))

        # ENSURE TO USE THE RIGHT SIGNAL (filtered or only flattened)
        (apex_sgn, base_sgn) = (apex_filt, base_filt) if param[pKeys.rank] is not 0 else (apex_flat, base_flat)

        ''' ===== [ FIND PEAKS, PERIODS and SELECT AP TO ANALYZE ] ===== '''

        # find peaks - restituisce (x, y) dei picchi
        # -> io passo il segnale invertito (perchè usa il 'max')
        # impongo che i picchi siano distanti fra loro almeno più di 2/3 del periodo
        # e prendo solo la x (tempo): [0]-> prendi solo la x del picco
        a_peaks = sign.find_peaks(-apex_sgn, distance=(2 / 3) * b[bKeys.AP_duration])[0]
        b_peaks = sign.find_peaks(-base_sgn, distance=(2 / 3) * b[bKeys.AP_duration])[0]

        # plotto i segnali con picchi e durate dei periodi
        plot_signals_with_peaks([apex_sgn, base_sgn], [a_peaks, b_peaks], ['Apex', 'Base'],
                                "Ramp at {}Hz".format(b[bKeys.freq_stim]),
                                _enum_peaks=True, _plot_intervals=True,
                                frame_rate=param[pKeys.frame_rate])

        # stima durata di ogni AP da differenza picchi consecutivi nell'apice
        print("- Control of stimulation frequency in the apex signal... ")
        AP_periods = np.diff(a_peaks) / param[pKeys.frame_rate]
        freq_stim_estimated = 1 / np.mean(AP_periods / 1000)
        print("-- Stimulation Frequency obtained: {0:0.1f}Hz".format(freq_stim_estimated))

        # user can select which potentials use to estimmate mean conduction velocity
        selection = input("\n***AP selection to estimate mean current velocity.\n"
                          "  Insert AP indexex between spaces. Example:\n"
                          "  0 2 3 5 7 8 9 [<- then press 'Enter'].\n"
                          "  Please, enter your selection here and the press 'Enter': \n")
        # extract selected indexes
        ap_selected_idx = selection.split(' ')
        print('- AP selected for Conduction Velocity Estimation: ')
        for l in ap_selected_idx:
            print(l, end='°, ')

        ''' ===== [ ANALYZE EACH ACTION POTENTIALS TO FIND DELAY APEX-BASE ] ===== '''
        # USE INTERPOLATION AND CROSS-CORRELATION TO ESTIMATE DELAY
        # from selected AP potentials
        # Idea: per ogni picco, prendo l'intorno giusto per selezionare il potenziale d'azione,
        # interpolo, calcolo delay usando il picco della cross-correlazione e poi medio tutti i delay

        cv_list = list()  # list of conduction velocity extracted for each AP

        # for each AP selected
        for (i_spike, spike_num) in enumerate(np.asarray(ap_selected_idx).astype(np.int)):

            # calculate extremes of selected action potential signal
            t1, t2   = np.int(b[bKeys.AP_duration] * spike_num), np.int(b[bKeys.AP_duration] * (spike_num + 1))
            ms_sel   = ms[t1:t2]        # time
            base_sel = base_sgn[t1:t2]  # base
            apex_sel = apex_sgn[t1:t2]  # apex

            # interpolo i due segnali di fattore 0.2 (come su LabView)
            dt = 0.2

            # calcolo funzione di interpolazione delle due tracce
            f_apex = interpolate.interp1d(ms_sel, apex_sel)
            f_base = interpolate.interp1d(ms_sel, base_sel)

            # creo nuovo asse temporale
            ms_res = np.arange(ms_sel[0], ms_sel[-1], dt)

            # resample signals using interpolation functions calculated above
            apex_res = f_apex(ms_res)
            base_res = f_base(ms_res)

            # estimate delay by cross-correlation max values
            delay_ms = lag_finder_in_ms(apex_res, base_res, 1000 / dt)

            # estimate and save conduction velocity
            cv = param[pKeys.ROI_distance_mm] / delay_ms
            cv_list.append(cv)

            # write spike_num and cv
            sheet.write(i_spike + 7, (i_burst * (n_col_sheet_block + 1)) + 1, "{}".format(int(spike_num)))
            sheet.write(i_spike + 7, (i_burst * (n_col_sheet_block + 1)) + 2, "{0:0.2f}".format(cv))

        # estimate mean and std. error
        avg = np.mean(np.asarray(cv_list))
        sem = stats.sem(np.asarray(cv_list))

        # write mean and std. error into excel file
        sheet.write(17, (i_burst * (n_col_sheet_block + 1)) + 2, "{0:0.2f}".format(avg))
        sheet.write(18, (i_burst * (n_col_sheet_block + 1)) + 2, "{0:0.2f}".format(sem))

        # print results and average
        print("*** RESULTS:")
        print("- Conduction velocities in m/s:")
        for cv in cv_list:
            print("-- {0:0.2f} m/s".format(cv))

        print("\n- Average Conduction velocity: ", end='')
        print("-- {0:0.2f} +- {1:0.2f} m/s".format(avg, sem))

    # save excel file
    wb.save(out_path)
    print('\nOutput saved in:')
    print(out_path)
    print(' --------- Conduction Velocity Analyzer: Process finished. ---------\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate conduction velocity for each stimulation frequency.")
    parser.add_argument('-i',
                        '--input-filepath',
                        nargs='+',
                        help='Path of txt file of optical mapping record to analyze')
    main(parser)
