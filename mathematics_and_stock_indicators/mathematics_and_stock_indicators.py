"""
DOCSTRING
"""
import urllib
import matplotlib
import matplotlib.dates as mpl_dates
import matplotlib.ticker as mpl_ticker
import matplotlib.pyplot as pyplot
import mpl_finance
import numpy
import pylab

matplotlib.rcParams.update({'font.size': 9})

def bytes_date_to_number(fmt, encoding='utf-8'):
    """
    DOCSTRING
    """
    string_converter = mpl_dates.strpdate2num(fmt)
    def bytes_converter(b_variable):
        string = b_variable.decode(encoding)
        return string_converter(string)
    return bytes_converter

def swing_index_calculation(
    open_1,
    open_2,
    high_1,
    high_2,
    low_1,
    low_2,
    close_1,
    close_2,
    limit_move
    ):
    """
    DOCSTRING
    """
    def calculate_k(high_2, low_2, close_1):
        """
        DOCSTRING
        """
        x_variable = high_2-close_1
        y_variable = low_2-close_1
        if x_variable > y_variable:
            k_variable = x_variable
            print(k_variable)
            return k_variable
        else:
            k_variable = y_variable
            print(k_variable)
            return k_variable

    def calculate_r(high_2, close_1, low_2, open_1, limit_move):
        """
        DOCSTRING
        """
        x_variable = high_2-close_1
        y_variable = low_2-close_1
        z_variable = high_2-low_2
        print(x_variable)
        print(y_variable)
        print(z_variable)
        if z_variable < x_variable > y_variable:
            print('x wins')
            r_variable = (high_2-close_1)-(0.5*(low_2-close_1))+(0.25*(close_1-open_1))
            print(r_variable)
            return r_variable
        elif x_variable < y_variable > z_variable:
            print('y wins')
            r_variable = (low_2-close_1)-(0.5*(high_2-close_1))+(0.25*(close_1-open_1))
            print(r_variable)
            return r_variable
        elif x_variable < z_variable > y_variable:
            print('z wins')
            r_variable = (high_2-low_2)+(0.25*(close_1-open_1))
            print(r_variable)
            return r_variable

    r_value = calculate_r(high_2, close_2, low_2, open_1, limit_move)
    k_value = calculate_k(high_2, low_2, close_1)

def compute_macd(x_variable, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence)
    using a fast and slow exponential moving average
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = exponential_moving_average(x_variable, slow)
    emafast = exponential_moving_average(x_variable, fast)
    return emaslow, emafast, emafast - emaslow

def exponential_moving_average(values, window):
    """
    DOCSTRING
    """
    weights = numpy.exp(numpy.linspace(-1.0, 0.0, window))
    weights /= weights.sum()
    average = numpy.convolve(values, weights, mode='full')[:len(values)]
    average[:window] = average[window]
    return average

def graph_data(ticker_symbol, moving_average_1, moving_average_2):
    """
    Use this to dynamically pull a ticker_symbol.
    """
    try:
        print('Currently Pulling', ticker_symbol)
        url_to_visit = (
            'http://chartapi.finance.yahoo.com/instrument/1.0/'
            + ticker_symbol
            + '/chartdata;type=quote;range=10y/csv'
            )
        stock_file = []
        try:
            source_code = urllib.request.urlopen(url_to_visit).read().decode()
            split_source = source_code.split('\n')
            for each_line in split_source:
                split_line = each_line.split(',')
                if len(split_line) == 6:
                    if 'values' not in each_line:
                        stock_file.append(each_line)
        except Exception as exception:
            print(str(exception), 'failed to organize pulled data.')
    except Exception as exception:
        print(str(exception), 'failed to pull pricing data')
    try:
        date, close_price, high_price, low_price, open_price, volume = numpy.loadtxt(
            stock_file,
            delimiter=',',
            unpack=True,
            converters={0: bytes_date_to_number('%Y%m%d')}
            )
        x_variable = 0
        y_variable = len(date)
        new_array = []
        while x_variable < y_variable:
            append_line = (
                date[x_variable],
                open_price[x_variable],
                high_price[x_variable],
                low_price[x_variable],
                close_price[x_variable],
                volume[x_variable]
                )
            new_array.append(append_line)
            x_variable += 1
        average_1 = moving_average(close_price, moving_average_1)
        average_2 = moving_average(close_price, moving_average_2)
        starting_point = len(date[moving_average_2-1:])
        figure = pyplot.figure(facecolor='#07000d')
        axis_1 = pyplot.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4, axisbg='#07000d')
        mpl_finance.candlestick_ohlc(
            axis_1,
            new_array[-starting_point:],
            width=0.6,
            colorup='#53c156',
            colordown='#ff1717'
            )
        label_1 = str(moving_average_1)+' SMA'
        label_2 = str(moving_average_2)+' SMA'
        axis_1.plot(
            date[-starting_point:],
            average_1[-starting_point:],
            '#e1edf9',
            label=label_1,
            linewidth=1.5
            )
        axis_1.plot(
            date[-starting_point:],
            average_2[-starting_point:],
            '#4ee6fd',
            label=label_2,
            linewidth=1.5
            )
        axis_1.grid(True, color='w')
        axis_1.xaxis.set_major_locator(mpl_ticker.MaxNLocator(10))
        axis_1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%Y-%m-%d'))
        axis_1.yaxis.label.set_color("w")
        axis_1.spines['bottom'].set_color("#5998ff")
        axis_1.spines['top'].set_color("#5998ff")
        axis_1.spines['left'].set_color("#5998ff")
        axis_1.spines['right'].set_color("#5998ff")
        axis_1.tick_params(axis='y', colors='w')
        pyplot.gca().yaxis.set_major_locator(mpl_ticker.MaxNLocator(prune='upper'))
        axis_1.tick_params(axis='x', colors='w')
        pyplot.ylabel('Stock price and Volume')
        moving_average_legend = pyplot.legend(
            loc=9,
            ncol=2,
            prop={'size':7},
            fancybox=True,
            borderaxespad=0.0
            )
        moving_average_legend.get_frame().set_alpha(0.4)
        text_ed = pylab.gca().get_legend().get_texts()
        pylab.setp(text_ed[0:5], color='w')
        volume_minimum = 0
        axis_0 = pyplot.subplot2grid(
            (6, 4),
            (0, 0),
            sharex=axis_1,
            rowspan=1,
            colspan=4,
            axisbg='#07000d'
            )
        relative_strength_index = relative_strength_index_calculation(close_price)
        relative_strength_index_color = '#c1f9f7'
        positive_color = '#386d13'
        negative_color = '#8f2020'
        axis_0.plot(
            date[-starting_point:],
            relative_strength_index[-starting_point:],
            relative_strength_index_color,
            linewidth=1.5
            )
        axis_0.axhline(70, color=negative_color)
        axis_0.axhline(30, color=positive_color)
        axis_0.fill_between(
            date[-starting_point:],
            relative_strength_index[-starting_point:],
            70,
            where=(relative_strength_index[-starting_point:] >= 70),
            facecolor=negative_color,
            edgecolor=negative_color,
            alpha=0.5
            )
        axis_0.fill_between(
            date[-starting_point:],
            relative_strength_index[-starting_point:],
            30,
            where=(relative_strength_index[-starting_point:] <= 30),
            facecolor=positive_color,
            edgecolor=positive_color,
            alpha=0.5
            )
        axis_0.set_yticks([30, 70])
        axis_0.yaxis.label.set_color("w")
        axis_0.spines['bottom'].set_color("#5998ff")
        axis_0.spines['top'].set_color("#5998ff")
        axis_0.spines['left'].set_color("#5998ff")
        axis_0.spines['right'].set_color("#5998ff")
        axis_0.tick_params(axis='y', colors='w')
        axis_0.tick_params(axis='x', colors='w')
        pyplot.ylabel('RSI')
        axis_1_volume = axis_1.twinx()
        axis_1_volume.fill_between(
            date[-starting_point:],
            volume_minimum,
            volume[-starting_point:],
            facecolor='#00ffe8',
            alpha=0.4
            )
        axis_1_volume.axes.yaxis.set_ticklabels([])
        axis_1_volume.grid(False)
        axis_1_volume.set_ylim(0, 3*volume.max())
        axis_1_volume.spines['bottom'].set_color("#5998ff")
        axis_1_volume.spines['top'].set_color("#5998ff")
        axis_1_volume.spines['left'].set_color("#5998ff")
        axis_1_volume.spines['right'].set_color("#5998ff")
        axis_1_volume.tick_params(axis='x', colors='w')
        axis_1_volume.tick_params(axis='y', colors='w')
        axis_2 = pyplot.subplot2grid(
            (6, 4),
            (5, 0),
            sharex=axis_1,
            rowspan=1,
            colspan=4,
            axisbg='#07000d'
            )
        fillcolor = '#00ffe8'
        nema = 9
        _, _, macd = compute_macd(close_price)
        ema9 = exponential_moving_average(macd, nema)
        axis_2.plot(date[-starting_point:], macd[-starting_point:], color='#4ee6fd', lw=2)
        axis_2.plot(date[-starting_point:], ema9[-starting_point:], color='#e1edf9', lw=1)
        axis_2.fill_between(
            date[-starting_point:],
            macd[-starting_point:] - ema9[-starting_point:],
            0,
            alpha=0.5,
            facecolor=fillcolor,
            edgecolor=fillcolor
            )
        pyplot.gca().yaxis.set_major_locator(mpl_ticker.MaxNLocator(prune='upper'))
        axis_2.spines['bottom'].set_color("#5998ff")
        axis_2.spines['top'].set_color("#5998ff")
        axis_2.spines['left'].set_color("#5998ff")
        axis_2.spines['right'].set_color("#5998ff")
        axis_2.tick_params(axis='x', colors='w')
        axis_2.tick_params(axis='y', colors='w')
        pyplot.ylabel('MACD', color='w')
        axis_2.yaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=5, prune='upper'))
        for label in axis_2.xaxis.get_ticklabels():
            label.set_rotation(45)
        pyplot.suptitle(ticker_symbol.upper(), color='w')
        pyplot.setp(axis_0.get_xticklabels(), visible=False)
        pyplot.setp(axis_1.get_xticklabels(), visible=False)
        axis_1.annotate(
            'Big news!',
            (date[510], average_1[510]),
            xytext=(0.8, 0.9),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='white', shrink=0.05),
            fontsize=14,
            color='w',
            horizontalalignment='right',
            verticalalignment='bottom')
        pyplot.subplots_adjust(left=0.09, bottom=0.14, right=0.94, top=0.95, wspace=0.20, hspace=0)
        pyplot.show()
        figure.savefig('example.png', facecolor=figure.get_facecolor())
    except Exception as exception:
        print('main loop', str(exception))

def moving_average(values, window):
    """
    DOCSTRING
    """
    weigths = numpy.repeat(1.0, window)/window
    simple_moving_averages = numpy.convolve(values, weigths, 'valid')
    return simple_moving_averages

def relative_strength_index_calculation(prices, n_variable=14):
    """
    DOCSTRING
    """
    deltas = numpy.diff(prices)
    seed = deltas[:n_variable+1]
    up_value = seed[seed >= 0].sum()/n_variable
    down_value = -seed[seed < 0].sum()/n_variable
    relative_strength = up_value/down_value
    relative_strength_index = numpy.zeros_like(prices)
    relative_strength_index[:n_variable] = 100.0-100.0/(1.0+relative_strength)
    for i in range(n_variable, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta
        up_value = (up_value*(n_variable-1) + upval)/n_variable
        down_value = (down_value*(n_variable-1) + downval)/n_variable
        relative_strength = up_value/down_value
        relative_strength_index[i] = 100.0-100.0/(1.0+relative_strength)
    return relative_strength_index

if __name__ == '__main__':
    while True:
        STOCK = input('Stock to plot: ')
        graph_data(STOCK, 10, 50)
