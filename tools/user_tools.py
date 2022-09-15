# -*- coding: utf-8 -*-
# Useful classes/functions

import os, time, sys, re, threading, traceback, warnings
import numpy as np
from tools.ANSI_tools import color_text, ANSI_RESET

class Console:
    """
    A simple class for printing colorful messages to the terminal asynchronously. Messages printed to the terminal via the 'out' method are queued and then printed in order; this ensures that messages printed to the terminal from different threads don't accidentally print simultaneously and produce garbled text.
    """

    busy = False
    queue = []

    @classmethod
    def out(Console, text, prefix="", indent=0, color=None, newline=True, newline_at_start=False):
        """
        Queues text to be printed to the terminal.

        text             - (string) Text to print to the terminal.
        prefix           - (string) A string to be inserted before 'text'.
        indent           - (int) The number of spaces to insert before 'text'.
        color            - (hexcode, or 3-tuple of integers) Color the text should print with.
        newline          - (boolean) If True, prints each string passed to Console.out on its own
                           line, sequentially.
        newline_at_start - (boolean) Set to True to keep the cursor at the end of the line instead
                           of sending it to the next line. This is useful if you need to frequently
                           erase/overwrite a line that has just been printed.
        """
        Console.queue.insert(0, {
            "text": text,
            "prefix": prefix,
            "indent": indent,
            "color": color,
            "newline": newline,
            "newline_at_start": newline_at_start
        })
        Console.clear_queue()

    @classmethod
    def clear_queue(Console):
        """
        Prints all the text currently stored in the queue to the terminal, then clears the queue.
        Any other calls to clear_queue while this function is running will do nothing, since only
        one function call is necessary to clear the entire queue.
        """
        if Console.busy:
            return

        def write_txt(text):
            sys.stdout.write( text )
            # with open("log.txt","w") as f:
            #     f.write(text)

        Console.busy = True

        while len(Console.queue) > 0:
            try:
                last_item = Console.queue.pop()
            except:
                break

            text, prefix, indent, color, newline, newline_at_start = ( last_item[key] for key in ("text","prefix","indent","color","newline","newline_at_start") )

            if newline and newline_at_start:
                write_txt("\n")
            indentation = indent * "    "
            write_txt( indentation+str(prefix)+color_text(str(text), fg=color) )
            if newline and not newline_at_start:
                write_txt("\n")

        Console.busy = False

    # Some terminal message presets for error logging.
    @classmethod
    def warn(Console, text, prefix="", indent=0):
        Console.out("WARNING: "+text, prefix=prefix, indent=indent, color="red")
    @classmethod
    def note(Console, text, prefix="", indent=0):
        Console.out("NOTE: "+text, prefix=prefix, indent=indent, color="yellow")
    @classmethod
    def newline(Console):
        Console.out("", newline=True)
    @classmethod
    def erase(Console):
        Console.out("\033[F\033[K", newline=False)

    class Spinner:
        """
        A simple subclass that creates a animated 'loading icon' that runs while a function is
        doing work in the background. This subclass is designed to be used with the 'task' function.
        """
        run = False
        delay = 0.2 # default seconds per frame for the animation.

        icons = {
            "dots": [" ⠋"," ⠙"," ⠚"," ⠞"," ⠖"," ⠦"," ⠴"," ⠲"," ⠳"," ⠓"]
            ,"bar": [" |"," /"," -"," \\"]
            ,"ellipsis": [ "."*(N) + " "*(6-N) for N in range(6) ]
        }

        @staticmethod
        def spinning_cursor(icon):
            while 1:
                for cursor in Console.Spinner.icons[icon]: yield cursor

        def __init__(self, time_limit=np.inf, delay=None, color="cyan", icon="ellipsis", text="", indent=0, attempt_number=1, print_attempts=False, no_spinner=False, no_print=False):
            self.no_print = no_print
            self.no_spinner = no_spinner
            self.attempt_number = attempt_number
            self.print_attempts = print_attempts
            self.indent = indent
            self.text = text
            self.time_limit = time_limit
            self.color = color
            self.spinner_generator = self.spinning_cursor(icon)
            if delay and float(delay): self.delay = delay

        def update_text(self, text):
            """
            Updates the text that precedes the spinner.
            """
            self.text = text
            if self.no_spinner and not self.no_print:
                Console.out(self.text+"...... "+str(round(self.Dt, 2))+"s", color=self.color, indent=self.indent)

        def spinner_task(self):
            """
            Render loop for the spinner animation.
            """
            Timer.click("spinner")
            time_limit_exceeded = False
            try:
                while self.run:

                    Dt = Timer.click("spinner", reset=False, format=False)
                    self.Dt = Dt
                    if Dt >= self.time_limit:
                        Console.erase()
                        if not self.no_print:
                            Console.warn("Task exceeded time limit of "+str(round(self.time_limit,2))+"s.")
                        time_limit_exceeded = True
                        break

                    Console.busy = True

                    if self.print_attempts:
                        spinner_frame = "    "*self.indent+self.text + ": "+str(self.attempt_number)+" attempts remaining" + next(self.spinner_generator) + " " + str(round(Dt, 2))+"s"
                    else:
                        spinner_frame = "    "*self.indent+self.text + next(self.spinner_generator) + " " + str(round(Dt, 2))+"s"

                    sys.stdout.write( color_text( spinner_frame, fg=self.color ) )
                    sys.stdout.write( ANSI_RESET )
                    sys.stdout.flush()

                    time.sleep(self.delay)

                    sys.stdout.write( '\033[F\n\033[K' )
                    # sys.stdout.write( '\b'*len(spinner_frame) )
                    sys.stdout.flush()

                    Console.busy = False

                    Console.clear_queue()
            except (KeyboardInterrupt, SystemExit):
                raise
                sys.exit()
            except:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info

            if time_limit_exceeded:
                raise RuntimeError

            Timer.click("spinner", reset=True)

        def __enter__(self):
            self.run = True
            if not self.no_print:
                if self.no_spinner:
                    Console.out(self.text, color="green", indent=self.indent)
                else:
                    threading.Thread(target=self.spinner_task).start()
            return self.update_text

        def __exit__(self, exception, value, tb):
            self.run = False
            time.sleep(self.delay)
            if exception is not None:
                return False

class Timer:
    """
    A simple class for timing functions.
    """
    timers = {}
    @classmethod
    def click(Timer, key, format=True, reset=True):
        """
        Starts and stops the timer.

        key    - (string) The name of the timer. A timer started with "Timer.click('sample-key')"
                 can only be stopped with a second call "Timer.click('sample-key')".
        format - (boolean) If False, Timer.click returns the number of seconds timed as a float. If
                 True, Timer.click returns the number of seconds as a string, ending with an 's'
                 for seconds (e.g. '32.05s').
        reset  - (boolean) If False, returns the time accululated without reseting the timer.
        """
        tf = time.time()
        if key in Timer.timers.keys():
            if reset:
                ti = Timer.timers[key]
                Timer.timers[key] = time.time()
            else:
                ti = Timer.timers[key]
            Dt = tf - ti
            if format:
                return str(round(Dt, 2))+"s"
            else:
                return Dt
        else:
            Timer.timers[key] = time.time()
            if format:
                return str(round(0, 2))+"s"
            else:
                return 0
    @classmethod
    def key_exists(Timer, key):
        return key in Timer.timers.keys()

def task(
    func,
    start_text="Beginning task",
    end_text="Finished task",
    fail_text="Failed task",
    indent=0,
    overwrite_line=True,
    exit_on_fail=True,
    print_traceback=True,
    number_of_attempts=1,
    time_limit=np.inf, # Note that this isn't implemented yet. If the set time limit is exceeded, you'll get all the associated error messages but the task won't cease. To make the task cease, the thread running the Spinner needs to be able to communicate to the thread running the task.
    pass_update_text=False,
    print_attempts=False,
    no_spinner=False,
    return_time=False,
    no_print=False,
    suppress_warnings=[]
    ):
    """
    A simple wrapper for functions that take a long time to run. The task wrapper provides several
    options:
        - Functions can be run with an animated 'spinner' that prints to the terminal (along with
        the amount of time the function has been running for) and indicates that Python is still
        running.
        - Long running functions that occasionally fail can be given multiple attempts to succeed.
        - The function can report its current status to the terminal via the 'update_text' function.

    e.g.
    def generate_rats(n):
        g = generate_rationals() # a generator for rational numbers
        for i, rat in zip(range(int(n)), g):
            if rat[0]==1:
                Console.out( str(i) + ": " + str(rat) )

    task(
        generate_rats
        ,start_text="Generating rational numbers"
        ,end_text="Finished generating rational numbers"
        ,fail_text="Failed to generate rational numbers"
        ,exit_on_fail=False
    )( 1e7 )
    """

    number_of_attempts = { "num": number_of_attempts } # This is a cheap hack that makes number_of_attempts accessible in the scope of the function f.

    def f(*args, **kwargs):
        # A function wrapper that adds a spinner as the function runs and times the function.
        Timer.click("task")

        with warnings.catch_warnings():
            for warning in suppress_warnings:
                warnings.filterwarnings('ignore', warning)

            while number_of_attempts["num"] > 0:
                try:
                    with Console.Spinner(time_limit=time_limit, text=start_text, indent=indent, attempt_number=number_of_attempts["num"], print_attempts=print_attempts, no_spinner=no_spinner, no_print=no_print) as update_text:
                        if pass_update_text:
                            kwargs["update_text"] = update_text
                        output = func(*args, **kwargs)
                        break #break while loop
                except (KeyboardInterrupt, SystemExit):
                    raise
                    sys.exit()
                except:
                    if print_traceback:
                        exc_info = sys.exc_info()
                        traceback.print_exception(*exc_info)
                        del exc_info

                    number_of_attempts["num"] -= 1
                    if number_of_attempts["num"] == 0:
                        Dt = Timer.click("task")
                        Console.newline()
                        Console.warn(fail_text+" in "+Dt+".", indent=indent)

                        if exit_on_fail:
                            Console.warn("Exiting program.", indent=indent)
                            sys.exit()
                        else:
                            if return_time:
                                return (None, Dt)
                            else:
                                return None

            Dt = Timer.click("task")
            if overwrite_line:
                Console.newline()
                Console.erase()

            if not no_print:
                if type(end_text) == type(f):
                    text = end_text(output, Dt)
                    Console.out(text, color="green", indent=indent)
                else:
                    Console.out(end_text+" in "+Dt+".", color="green", indent=indent)

            if return_time:
                return ( output, Dt )
            else:
                return output
    return f

def save_table_to_txt( filename, table, column_titles=None, column_padding=3 ):
    """
    Saves a python dictionary to a txt file formatted as a table.

    filename        - (string) The name of the txt file to save the table to. (e.g. 'data.txt')
    table           - (dict) A dictionary where each key-value pair is the column title and a list
                      of values from each row.
    column_titles   - (list of strings) An ordered list of the column titles.
    column_padding  - (int) The number of spaces to insert between each column.
    """

    def padstr(string, length):
        return string+" "*(length-len(string))

    table_keys = table.keys()
    if column_titles is None:
        column_titles = list(table.keys())

    column_widths = []
    for column_title in column_titles:
        assert column_title in table_keys, f"{column} is not a key of table."
        column_width = max([ max([ len(str(e)) for e in table[column_title] ]), len(str(column_title)) ])
        column_widths.append( column_width )

        with open(filename,"w") as f:

            # Write column titles
            f.write( (" "*column_padding).join( padstr(str(col_title),col_width) for col_title, col_width in zip(column_titles,column_widths) )+"\n" )

            # Write lines
            table_length = len(table[column_titles[0]])
            for i in range(table_length):

                cells = [ str( table[column_title][i] ) for column_title in column_titles ]
                f.write( (" "*column_padding).join( padstr(cell,col_width) for cell, col_width in zip(cells,column_widths) ) + "\n" )

def read_table_from_txt(filename, column_padding=3, column_processes=None):
    """
    Loads data from a txt file saved using save_table_to_txt. Returns a dictionary where each key
    is the column title, and each key is associated with a list of values from that column.

    filename          - (string) The path to the txt file to load.
    column_padding    - (int) The number of spaces between each column.
    column_processes  - (list) A list of functions to apply to each column. If any element of this
                        array is 'None', each value in the associated column will be interpreted as
                        a string.

    output            - (dict) A dictionary where each key is the title of a column and the
                        associated value is a list of cell content from each row in the column.
    """
    table = {}
    column_spacer = re.compile("[ ]{"+str(column_padding)+",}")

    with open(filename,"rt") as f:
        need_column_titles = True
        need_column_processes = True
        for line in f.readlines():
            line = line.split("\n")[0]

            if need_column_titles:
                # initialize table with column titles as keys
                column_titles = column_spacer.split(line)
                column_titles = [ column_title for column_title in column_titles if column_title != '' ]
                for column_title in column_titles:
                    table[column_title] = []
                need_column_titles = False
                continue

            if need_column_processes:
                if column_processes is None:
                    column_processes = [ (lambda x: x) for _ in range(len(column_titles)) ]
                else:
                    assert len(column_processes) == len(column_titles), "The number of column processing functions does not match the number of columns."
                    for i in range(len(column_titles)):
                        if column_processes[i] is None:
                            column_processes[i] = (lambda x: x)
                need_column_processes = False

            values = column_spacer.split(line)
            values = [ value for value in values if value != '' ]
            for value, column_title, column_process in zip(values,column_titles,column_processes):
                table[column_title].append( column_process(value) )
    return table

def concatenate_tables(*tables):
    # tables must have the same columns
    column_titles = tables[0].keys()
    new_table = { column_title: [] for column_title in column_titles }
    for column_title in column_titles:
        new_table[column_title] = []
        for table in tables:
            new_table[column_title] += table[column_title]
    return new_table

def print_dict(d, print_func=print):
    """
    Prints dictionaries such that more nested data receive more indentation.

    d           - (dict) A python dictionary.
    print_func  - (function) A function that accepts a string as input to print to the terminal. 
    """

    def quote_str(el):
        if type(el) is str:
            return "'"+el+"'"
        else:
            return str(el)

    def pd(d, indent=1, ofe=False, top_level=True, print_func=print_func):
        if top_level:
            print_func("{")
        keys = d.keys()
        len_keys = len(keys)
        for i, key in enumerate(keys):
            on_final_element = i == len_keys-1
            if type(d[key]) is not type({}):
                line =  f"{'   '*indent}{ quote_str(key) }: { quote_str(d[key]) }" + ( "" if on_final_element else "," )
                print_func( line )
            else:
                print_func( f"{'   '*indent}{ quote_str(key) }:" + " {" )
                pd( d[key], indent=indent+1, ofe=on_final_element, top_level=False )
        if top_level:
            print_func("}")
        else:
            print_func( ('   '*(indent-1))+"}" + ( "" if ofe else "," ) )
    pd(d)

def subset_colormap(original_cmap, new_min=0, new_max=1):
    """
    Creates a matplotlib colormap from a subset of an existing matplotlib colormap.

    original_cmap   - (colormap) The matplotlib colormap to be sampled from.
    new_min         - (float/int, ranging from 0-1) The 'start' of the subset.
    new_max         - (float/int, ranging from 0-1) The 'end' of the subset.

    output          - (colormap)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcol
    cmap = plt.get_cmap(original_cmap)
    new_map = mcol.LinearSegmentedColormap.from_list(
            original_cmap + '_subset',
            cmap(np.linspace(new_min, new_max, 100))
        )
    return new_map
