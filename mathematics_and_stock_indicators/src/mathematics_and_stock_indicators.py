"""
DOCSTRING
"""
import datetime
import urllib
import math
import matplotlib
import matplotlib.dates as mpl_dates
import matplotlib.ticker as mpl_ticker
import matplotlib.pyplot as pyplot
import mpl_finance
import numpy
import pylab
import time

matplotlib.rcParams.update({'font.size': 9})

LIMIT_MOVE = 75

class AverageTrueRange:

    def __init__(
            self,
            dates,
            close_prices,
            high_prices,
            low_prices,
            open_prices,
            timeframe):
        """
        DOCSTRING
        """
        true_range_dates = []
        true_ranges = []
        for count, element in enumerate(dates, 1):
            true_range_date, true_range = self.true_range(
                dates[count],
                close_prices[count],
                high_prices[count],
                low_prices[count],
                open_prices[count],
                close_prices[count-1]
                )
            true_range_dates.append(true_range_date)
            true_ranges.append(true_range)
        return GraphData().exponential_moving_average(true_ranges, 14)

    def true_range(
            self,
            date,
            high_price,
            low_price,
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
        return date, true_range

class GraphData:
    """
    DOCSTRING
    """
    def bytes_date_to_number(self, fmt, encoding='utf-8'):
        """
        DOCSTRING
        """
        string_converter = mpl_dates.strpdate2num(fmt)
        def bytes_converter(b_variable):
            string = b_variable.decode(encoding)
            return string_converter(string)
        return bytes_converter

    def exponential_moving_average(self, values, window):
        """
        Calculate exponential moving average.
        """
        weights = numpy.exp(numpy.linspace(-1.0, 0.0, window))
        weights /= weights.sum()
        average = numpy.convolve(values, weights, mode='full')[:len(values)]
        average[:window] = average[window]
        return average

    def graph_data(self, ticker_symbol, moving_average_1, moving_average_2):
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
                converters={0: self.bytes_date_to_number('%Y-%m-%d %H:%M:%S')}
                )
            new_array = []
            for count, element in enumerate(date):
                append_line = (
                    element,
                    open_price[count],
                    high_price[count],
                    low_price[count],
                    close_price[count],
                    volume[count]
                    )
                new_array.append(append_line)
            average_1 = self.simple_moving_average(close_price, moving_average_1)
            average_2 = self.simple_moving_average(close_price, moving_average_2)
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
            date, top_band, bottom_band, middle_band = StockIndicators().bollinger_bands(2, 20)
            axis_1.plot(date[-starting_point:], top_band[-starting_point:], '#C1F9F7', alpha=0.7)
            axis_1.plot(date[-starting_point:], bottom_band[-starting_point:], '#C1F9F7', alpha=0.7)
            axis_1.plot(date[-starting_point:], middle_band[-starting_point:], '#C1F9F7', alpha=0.7)
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
            relative_strength_index = self.relative_strength_index(close_price)
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
            _, _, macd = self.macd(close_price)
            ema9 = self.exponential_moving_average(macd, nema)
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
            for count, element in enumerate(date[1:], 1):
                try:
                    y_variable = StockIndicators().swing_index(
                        open_price[count-1],
                        open_price[count],
                        high_price[count],
                        low_price[count],
                        close_price[count-1],
                        close_price[count],
                        LIMIT_MOVE
                        )
                    swing_index_y.append(y_variable)
                    swing_index_date.append(element)
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

    def macd(self, x_variable, slow=26, fast=12):
        """
        compute the MACD (Moving Average Convergence/Divergence)
        using a fast and slow exponential moving average
        return value is emaslow, emafast, macd which are len(x) arrays
        """
        ema_slow = self.exponential_moving_average(x_variable, slow)
        ema_fast = self.exponential_moving_average(x_variable, fast)
        return ema_slow, ema_fast, ema_fast - ema_slow

    def simple_moving_average(self, values, window):
        """
        Calculate moving average.
        """
        weigths = numpy.repeat(1.0, window)/window
        simple_moving_averages = numpy.convolve(values, weigths, 'valid')
        return simple_moving_averages

    def relative_strength_index(self, prices, n_variable=14):
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
    def __init__(self):
        sample_data = open('data/sample_data.txt', 'r').read()
        split_data = sample_data.split('\n')
        self.date,\
            self.close_price,\
            self.high_price,\
            self.low_price,\
            self.open_price,\
            _ = numpy.loadtxt(
                split_data,
                delimiter=',',
                unpack=True
                )

    def aroon(self, timeframe):
        """
        Calculate Aroon.
        """
        aroon_ups = []
        aroon_downs = []
        aroon_dates = []
        aroon_oscillations = []
        for x_variable in range(timeframe, len(self.date)):
            aroon_up = ((self.high_price[x_variable-timeframe:x_variable].to_list().index(
                max(self.high_price[x_variable-timeframe:x_variable])
                ))/float(timeframe))*100
            aroon_down = ((self.low_price[x_variable-timeframe:x_variable].to_list().index(
                max(self.low_price[x_variable-timeframe:x_variable])
                ))/float(timeframe))*100
            aroon_oscillation = aroon_up-aroon_down
            aroon_oscillations.append(aroon_oscillation)
            aroon_ups.append(aroon_up)
            aroon_downs.append(aroon_down)
            aroon_dates.append(self.date[x_variable])
        return aroon_dates, aroon_ups, aroon_downs, aroon_oscillations

    def average_directional_index(self):
        """
        Calculate average directional index.
        """
        positive_directional_index, \
            negative_directional_index = self.directional_indices()
        z_variable = 0
        directional_indices = []
        while z_variable < len(self.date[1:]):
            directional_index = 100*(
                abs(positive_directional_index[z_variable]-negative_directional_index[z_variable])/
                (positive_directional_index[z_variable]+negative_directional_index[z_variable])
                )
            directional_indices.append(directional_index)
            z_variable += 1
        average_directional_index = GraphData().exponential_moving_average(directional_indices, 14)
        print(average_directional_index)

    def bollinger_bands(self, multiplier, timeframe):
        """
        DOCSTRING
        """
        band_dates = []
        top_bands = []
        bottom_bands = []
        middle_bands = []
        for count, element in enumerate(self.date, timeframe):
            current_simple_moving_average = GraphData().simple_moving_average(
                self.close_price[count-timeframe:count], 
                timeframe
                )[-1]
            _, current_standard_deviation = self.standard_deviation(
                timeframe, 
                self.close_price[0:timeframe]
                )
            current_standard_deviation = current_standard_deviation[-1]
            top_band = current_simple_moving_average+(current_standard_deviation*multiplier)
            bottom_band = current_simple_moving_average-(current_standard_deviation*multiplier)
            band_dates.append(element)
            top_bands.append(top_band)
            bottom_bands.append(bottom_band)
            middle_bands.append(current_simple_moving_average)
        return band_dates, top_bands, bottom_bands, middle_bands

    def center_of_gravity(self, dates, data, timeframe):
        """
        DOCSTRING
        """
        center_of_gravities = []
        for count, element_x in enumerate(dates):
            consider = data[count-timeframe:count]
            multipliers = range(1, timeframe+1)
            numerator, denominator = 0, 0
            reversed_order = reversed(consider)
            ordered = []
            for element_y in reversed_order:
                ordered.append(element_y)
            for multiplier in multipliers:
                add_me = multiplier*ordered[multiplier-1]
                add_me_2 = ordered[multiplier-1]
                numerator += add_me
                denominator += add_me_2
            center_of_gravity = -(numerator/float(denominator))
            center_of_gravities.append(center_of_gravity)
        return dates[timeframe:], center_of_gravities

    def chaikin_money_flow(
            self, 
            date, 
            close_price, 
            high_price, 
            low_price, 
            open_price, 
            volume, 
            timeframe):
        """
        DOCSTRING
        """
        chaikin_money_flows = []
        money_flow_multipliers = []
        money_flow_volumes = []
        for count, element_x in enumerate(date, timeframe):
            period_volume = 0
            volume_range = volume[count-timeframe:count]
            for element_y in volume_range:
                period_volume += element_y
            money_flow_multiplier = (
                (close_price[count]-low_price[count])-(high_price[count]-close_price[count])
                 )/(high_price[count]-low_price[count])
            money_flow_volume = money_flow_multiplier*period_volume
            money_flow_multipliers.append(money_flow_multiplier)
            money_flow_volumes.append(money_flow_volume)
        for count in enumerate(money_flow_volumes, timeframe):
            period_volume = 0
            volume_range = volume[count-timeframe:count]
            for element_y in volume_range:
                period_volume += element_y
            consider = money_flow_volumes[count-timeframe:count]
            timeframes_money_flow_volume = 0
            for element in consider:
                timeframes_money_flow_volume += element
            timeframes_chaikin_money_flow = timeframes_money_flow_volume/period_volume
            chaikin_money_flows.append(timeframes_chaikin_money_flow)
        return date[timeframe+timeframe:], chaikin_money_flows

    def chaikin_volatility(self, ema_used, periods_ago):
        """
        DOCSTRING
        """
        chaikin_volatilities = []
        highs_minus_lows = []
        for count, element in enumerate(self.date):
            high_minus_low = self.high_price[count]-self.low_price[count]
            highs_minus_lows.append(high_minus_low)
        high_minus_low_ema = GraphData().exponential_moving_average(highs_minus_lows, ema_used)
        y_variable = ema_used + periods_ago
        for count, element in enumerate(self.date, y_variable):
            chaikin_volatility = self.percent_change(
                high_minus_low_ema[count-periods_ago], 
                high_minus_low_ema[count]
                )
            chaikin_volatilities.append(chaikin_volatility)
        return self.date[y_variable:], chaikin_volatilities

    def chande_momentum_oscillator(self, prices, timeframe):
        """
        DOCSTRING
        """
        chande_momentum_oscillators = []
        for count_x, element in enumerate(prices, timeframe):
            consideration_prices = prices[count_x-timeframe:count_x]
            up_sum, down_sum = 0, 0 
            for count_y in range(1, timeframe):
                current_price = consideration_prices[count_y]
                previous_price = consideration_prices[count_y-1]
                if current_price >= previous_price:
                    up_sum += current_price-previous_price
                else:
                    down_sum += previous_price-current_price
            current_cmo = ((up_sum-down_sum)/float(up_sum+down_sum))*100.00
            chande_momentum_oscillators.append(current_cmo)
        return self.date[timeframe:], chande_momentum_oscillators

    def commodity_channel_index(
            self,
            date, 
            close_price, 
            high_price, 
            low_price, 
            open_price, 
            volume, 
            timeframe,
            simple_moving_average):
        """
        DOCSTRING
        """
        typical_prices = []
        mean_deviations = []
        commodity_channel_indices = []
        for count, element in enumerate(high_price):
            typical_price = (high_price[count]+low_price[count]+close_price[count])/3
            typical_prices.append(typical_price)
        sma_typical_prices = GraphData().simple_moving_average(
            typical_prices, 
            simple_moving_average
            )
        typical_prices = typical_prices[simple_moving_average-1:]
        for count_a, element in enumerate(sma_typical_prices, timeframe):
            typical_prices_considered = typical_prices[count_a-timeframe:count_a]
            sma_typical_prices_considered = sma_typical_prices[count_a-timeframe:count_a]
            mean_deviation_sum = 0
            for count_b, element in enumerate(sma_typical_prices_considered, timeframe):
                mean_deviation = abs(
                    typical_prices_considered[count_b]-sma_typical_prices_considered[count_b]
                    )
                mean_deviation_sum += mean_deviation
            mean_deviations.append(mean_deviation_sum/timeframe)
        typical_prices = typical_prices[14:]
        sma_typical_prices = sma_typical_prices[14:]
        for count in range(0, sma_typical_prices):
            commodity_channel_indices.append(
                (typical_prices[count]-sma_typical_prices[count])/(0.015*mean_deviations[count])
            )
        return self.date[timefram+simple_moving_average-1:], commodity_channel_indices

    def directional_indices(self):
        """
        Calculate directional indices.
        """
        true_range_dates = []
        true_ranges = []
        positive_directional_movements = []
        negative_directional_movements = []
        for x_variable in range(1, len(self.date)):
            true_range_date, true_range = true_range(
                self.date[x_variable],
                self.close_price[x_variable],
                self.high_price[x_variable],
                self.low_price[x_variable],
                self.open_price[x_variable],
                self.close_price[x_variable-1]
                )
            true_range_dates.append(true_range_date)
            true_ranges.append(true_range)
            _, \
                positive_directional_movement, \
                negative_directional_movement = self.directional_movement(
                    self.date[x_variable],
                    self.high_price[x_variable],
                    self.low_price[x_variable],
                    self.open_price[x_variable-1],
                    self.high_price[x_variable-1]
                    )
            positive_directional_movements.append(positive_directional_movement)
            negative_directional_movements.append(negative_directional_movement)
        exponential_positive_directional_movement = GraphData().exponential_moving_average(
            positive_directional_movements, 14
            )
        exponential_negative_directional_movement = GraphData().exponential_moving_average(
            negative_directional_movements, 14
            )
        average_true_ranges = GraphData().exponential_moving_average(true_ranges, 14)
        y_variable = 1
        positive_directional_movements = []
        negative_directional_movements = []
        while y_variable < len(average_true_ranges):
            positive_directional_movement = 100*(
                exponential_positive_directional_movement[y_variable]/
                average_true_ranges[y_variable]
                )
            positive_directional_movements.append(positive_directional_movement)
            negative_directional_movement = 100*(
                exponential_negative_directional_movement[y_variable]/
                average_true_ranges[y_variable]
                )
            negative_directional_movements.append(negative_directional_movement)
            y_variable += 1
        return positive_directional_movements, negative_directional_movements

    def directional_movement(
            self,
            date,
            high_price,
            low_price,
            yesterdays_high_price,
            yesterdays_low_price):
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

    def ease_of_movement(
            self,
            date, 
            close_prices, 
            high_prices, 
            low_prices, 
            open_prices, 
            volumes, 
            timeframe):
        """
        DOCSTRING
        """
        one_period_emvs = []
        for count, element in enumerate(close_prices, 1):
            movement = ((high_prices[count]+low_prices[count])/2)-\
                ((high_prices[count-1]+low_prices[count-1])/2)
            box_rate = (volumes[count]/1000000.00)/(high_prices[count]-low_prices[count])
            one_period_emvs.append(movement/box_rate)
        emv_timeframe = GraphData().simple_moving_average(one_period_emvs, timeframe)
        return self.date[timeframe:], emv_timeframe

    def elder_force_index(
            self,
            date, 
            close_prices,
            volumes, 
            timeframe):
        """
        DOCSTRING
        """
        elder_force_indices = []
        for count, element in enumerate(date, 1):
            eelder_force_indices.append((close_prices[count]-close_prices[count-1])*volumes[count])
        timeframe_efi = GraphData().simple_moving_average(elder_force_indices, timeframe)
        return self.dates[1:], timeframe_efi

    def gopalakrishnan_range_index(
            self,
            dates,
            high_prices,
            low_prices,
            timeframe):
        """
        DOCSTRING
        """
        gopalakrishnan_range_indices = []
        for count, element in enumerate(dates, timeframe):
            highs_considered = high_prices[count-timeframe:count]
            lows_considered = low_prices[count-timeframe:count]
            highest_high = max(highs_considered)
            lowest_low = min(lows_considered)
            gopalakrishnan_range_index.append(
                math.log(highest_high-lowest_low)/math.log(timeframe)
                )
        return dates[timeframe:], gopalakrishnan_range_indices

    def highest_high_lowest_low(
            self,
            dates,
            close_prices,
            timeframe):
        """
        DOCSTRING
        """
        highest_highs = []
        lowest_lows = []
        for count, element in enumerate(dates, timeframe):
            values_considered = close_prices[count-timeframe:count]
            highest_highs.append(max(values_considered))
            lowest_lows.append(min(values_considered))
        return dates, highest_highs, lowest_lows

    def keltner_channels(
            self, 
            dates,
            close_prices,
            high_prices,
            low_prices,
            open_prices,
            volumes,
            timeframe_a, 
            timeframe_b
            ):
        """
        DOCSTRING
        """
        upper_line = []
        middle_line = []
        lower_line = []
        average_true_ranges = AverageTrueRange(
            dates,
            close_prices,
            high_prices,
            low_prices,
            open_prices,
            volumes,
            timeframe_b
            )
        timeframe_ema = GraphData().exponential_moving_average(close_prices, timeframe_a)
        timeframe_ema = timeframe_ema[1:]
        for count, element in enumerate(timeframe_ema):
            upper_lines.append(timeframe_ema[count]+(2*average_true_ranges[count]))
            middle_lines.append(timeframe_ema[count])
            lower_lines.append(timeframe_ema[count]-(2*average_true_ranges[count]))
        return upper_line, middle_line, lower_line

    def percent_change(self, start_point, current_point):
        """
        DOCSTRING
        """
        return ((float(current_point)-start_point)/abs(start_point))*100.00

    def standard_deviation(self, timeframe, prices):
        """
        DOCSTRING
        """
        standard_deviations = []
        standard_deviation_dates = []
        for count, element in enumerate(prices, timeframe):
            array_to_consider = prices[count-timeframe:count]
            standard_deviation = array_to_consider.std()
            standard_deviations.append(standard_deviation)
            standard_deviation_dates.append(element)
        return standard_deviation_dates, standard_deviations

    def swing_index(
            self,
            open_1,
            open_2,
            high_2,
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

        def calculate_r(high_2, close_1, low_2, open_1):
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
        r_value = calculate_r(high_2, close_2, low_2, open_1)
        k_value = calculate_k(high_2, low_2, close_1)
        numerator = close_2-close_1+(0.5*(close_2-open_2))+(0.25*(close_1-open_1))
        whole_fraction = numerator/r_value
        swing_index = 50*whole_fraction*(k_value/limit_move)
        return swing_index

if __name__ == '__main__':
    while True:
        STOCK = input('Stock to plot:')
        GraphData().graph_data(STOCK, 10, 50)
