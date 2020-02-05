"""
DOCSTRING
"""
import datetime
import urllib
import matplotlib
import matplotlib.dates as mpl_dates
import matplotlib.ticker as mpl_ticker
import matplotlib.pyplot as pyplot
import mpl_finance
import numpy
import pylab

matplotlib.rcParams.update({'font.size': 9})

LIMIT_MOVE = 75

class GraphData:

    def bytes_date_to_number(fmt, encoding='utf-8'):
        """
        DOCSTRING
        """
        string_converter = mpl_dates.strpdate2num(fmt)
        def bytes_converter(b_variable):
            string = b_variable.decode(encoding)
            return string_converter(string)
        return bytes_converter

    def exponential_moving_average(values, window):
        """
        Calculate exponential moving average.
        """
        weights = numpy.exp(numpy.linspace(-1.0, 0.0, window))
        weights /= weights.sum()
        average = numpy.convolve(values, weights, mode='full')[:len(values)]
        average[:window] = average[window]
        return average

    def graph_data(ticker_symbol, moving_average_1, moving_average_2):
        """
        Use this to dynamically pull a stock.
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
                    fix_me = split_line[0]
                    if len(split_line) == 6:
                        if 'values' not in each_line:
                            each_line = each_line.replace(
                                fix_me,
                                str(
                                    datetime.datetime.fromtimestamp(
                                        int(fix_me)
                                        ).strftime('%Y-%m-%d %H:%M:%S')
                                    )
                                )
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
                converters={0: bytes_date_to_number('%Y-%m-%d %H:%M:%S')}
                )
            new_array = []
            for x_variable in range(0, len(date)):
                append_line = (
                    date[x_variable],
                    open_price[x_variable],
                    high_price[x_variable],
                    low_price[x_variable],
                    close_price[x_variable],
                    volume[x_variable]
                    )
                new_array.append(append_line)
            average_1 = moving_average(close_price, moving_average_1)
            average_2 = moving_average(close_price, moving_average_2)
            starting_point = len(date[moving_average_2-1:])
            figure = pyplot.figure(facecolor='#07000d')
            axis_1 = pyplot.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4, axisbg='#07000d')
            mpl_finance.candlestick_ohlc(
                axis_1,
                new_array[-starting_point:],
                width=0.0006,
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
            relative_strength_index = relative_strength_index(close_price)
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
            _, _, macd = macd(close_price)
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
            swing_index_y = []
            swing_index_date = []
            for x_variable in range(1, len(date[1:])):
                try:
                    y_variable = swing_index_calculation(
                        open_price[x_variable-1],
                        open_price[x_variable],
                        high_price[x_variable-1],
                        high_price[x_variable],
                        low_price[x_variable-1],
                        low_price[x_variable],
                        close_price[x_variable-1],
                        close_price[x_variable],
                        LIMIT_MOVE
                        )
                    swing_index_y.append(y_variable)
                    swing_index_date.append(date[x_variable])
                except Exception as exception:
                    print(str(exception))
            axis_2.plot(swing_index_date, swing_index_y, 'w', linewidth=1.5)
            pyplot.ylabel('ATR(14)', negative_color='w')
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
            pyplot.subplots_adjust(
                left=0.09, 
                bottom=0.14, 
                right=0.94, 
                top=0.95, 
                wspace=0.20, 
                hspace=0
                )
            pyplot.show()
            figure.savefig('example.png', facecolor=figure.get_facecolor())
        except Exception as exception:
            print('main loop', str(exception))

    def macd(x_variable, slow=26, fast=12):
        """
        compute the MACD (Moving Average Convergence/Divergence)
        using a fast and slow exponential moving average
        return value is emaslow, emafast, macd which are len(x) arrays
        """
        ema_slow = exponential_moving_average(x_variable, slow)
        ema_fast = exponential_moving_average(x_variable, fast)
        return ema_slow, ema_fast, ema_fast - ema_slow

    def moving_average(values, window):
        """
        Calculate moving average.
        """
        weigths = numpy.repeat(1.0, window)/window
        simple_moving_averages = numpy.convolve(values, weigths, 'valid')
        return simple_moving_averages

    def relative_strength_index(prices, n_variable=14):
        """
        Calculate relative strength index.
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

class StockIndicators:
    """
    A collection of stock indicators.
    """
    def average_directional_index():
        """
        Calculate average directional index.
        """
        positive_directional_index, negative_directional_index = directional_indices()
        z_variable = 0
        directional_indices = []
        while z_variable < len(date[1:]):
            directional_index = 100*(
                abs(positive_directional_index[z_variable]-negative_directional_index[z_variable])/
                (positive_directional_index[z_variable]+negative_directional_index[z_variable])
                )
            directional_indices.append(directional_index)
            z_variable += 1
        average_directional_index = exponential_moving_average(directional_indices, 14)
        print(average_directional_index)

    def directional_indices():
        """
        Calculate directional indices.
        """
        true_range_dates = []
        true_ranges = []
        positive_directional_movements = []
        negative_directional_movements = []
        for x_variable in range(1, len(date)):
            true_range_date, true_range = true_range(
                date[x_variable],
                close_price[x_variable],
                high_price[x_variable],
                low_price[x_variable],
                open_price[x_variable],
                close_price[x_variable-1]
                )
            true_range_dates.append(true_range_date)
            true_ranges.append(true_range)
            directional_movement_date, \
                positive_directional_movement, \
                negative_directional_movement = directional_movement(
                    date[x_variable],
                    open_price[x_variable],
                    high_price[x_variable],
                    low_price[x_variable],
                    close_price[x_variable],
                    open_price[x_variable-1],
                    high_price[x_variable-1],
                    low_price[x_variable-1],
                    close_price[x_variable-1],
                    )
            positive_directional_movements.append(positive_directional_movement)
            negative_directional_movements.append(negative_directional_movement)
        exponential_positive_directional_movement = exponential_moving_average(
            positive_directional_movements, 14
            )
        exponential_negative_directional_movement = exponential_moving_average(
            negative_directional_movements, 14
            )
        average_true_ranges = exponential_moving_average(true_ranges, 14)
        y_variable = 1
        positive_directional_movements = []
        negative_directional_movements = []
        while y_variable < len(average_true_ranges):
            positive_directional_movement = 100*(
                exponential_positive_directional_movement[y_variable]/average_true_ranges[y_variable]
                )
            positive_directional_movements.append(positive_directional_movement)
            negative_directional_movement = 100*(
                exponential_negative_directional_movement[y_variable]/average_true_ranges[y_variable]
                )
            negative_directional_movements.append(negative_directional_movement)
            y_variable += 1
        return positive_directional_movements, negative_directional_movements

    def directional_movement(
        date,
        open_price,
        high_price,
        low_price,
        close_price,
        yesterdays_open_price,
        yesterdays_high_price,
        yesterdays_low_price,
        yesterdays_close_price):
        """
        Calculate directional movement.
        """
        move_up = high_price-yesterdays_high_price
        move_down = low_price-yesterdays_low_price
        if 0 < move_up > move_down:
            positive_directional_movement = move_up
        else:
            positive_directional_movement = 0
        if 0 < move_down > move_up:
            negative_directional_movement = move_down
        else:
            negative_directional_movement = 0
        return date, positive_directional_movement, negative_directional_movement

    def swing_index(
        open_1,
        open_2,
        high_1,
        high_2,
        low_1,
        low_2,
        close_1,
        close_2,
        limit_move):
        """
        Calculate swing index.
        """
        def calculate_k(high_2, low_2, close_1):
            """
            Calculate K value.
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
            Calculate R value.
            """
            x_variable = high_2-close_1
            y_variable = low_2-close_1
            z_variable = high_2-low_2
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
        top_fraction = close_2-close_1+(0.5*(close_2-open_2))+(0.25*(close_1-open_1))
        whole_fraction = top_fraction/r_value
        swing_index = 50*whole_fraction*(k_value/limit_move)
        return swing_index

    def true_range(
        date, 
        close_price, 
        high_price, 
        low_price, 
        open_price, 
        yesterdays_close_price):
        """
        Calculate true range.
        """
        x_variable = high_price-low_price
        y_variable = abs(high_price-yesterdays_close_price)
        z_variable = abs(low_price-yesterdays_close_price)
        if y_variable <= x_variable >= z_variable:
            true_range = x_variable
        elif x_variable <= y_variable >= z_variable:
            true_range = y_variable
        elif x_variable <= z_variable >= y_variable:
            true_range = z_variable
        print(date, true_range)
        return date, true_range
    true_range_dates = []
    true_ranges = []
    for x_variable in range(1, len(date)):
        true_range_date, true_range = true_range(
            date[x_variable], 
            close_price[x_variable], 
            high_price[x_variable], 
            low_price[x_variable], 
            open_price[x_variable], 
            close_price[x_variable-1]
            )
        true_range_dates.append(true_range_date)
        true_ranges.append(true_range)
        print(true_range)

if __name__ == '__main__':
    sample_data = open('data/sample_data.txt', 'r').read()
    split_data = sample_data.split('\n')
    date, close_price, high_price, low_price, open_price, volume = numpy.loadtxt(
    split_data, 
    delimiter=',', 
    unpack=True
    )
    while True:
        STOCK = input('Stock to plot:')
        graph_data(STOCK, 10, 50)
