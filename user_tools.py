# -*- coding: utf-8 -*-
# Useful classes/functions

import os, time, sys, re, threading, traceback, warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Console:
    """
    A simple class for printing colorful messages to the terminal asynchronously. Messages printed to the terminal via the 'out' method are queued and then printed in order; this ensures that messages printed to the terminal from different threads don't accidentally print simultaneously and produce garbled text.
    """

    colors = {
        "RED": "\033[1;31m"
        ,"BLUE": "\033[1;34m"
        ,"CYAN": "\033[1;36m"
        ,"GREEN": "\033[0;32m"
        ,"RESET": "\033[0;0m"
        ,"BOLD": "\033[;1m" #makes things yellow in powershell
        ,"REVERSE": "\033[;7m"
    }

    busy = False
    queue = []

    @classmethod
    def out(Console, text, prefix="", indent=0, decoration=None, newline=True, newline_at_start=False):
        """
        Queues text to be printed to the terminal.

        text             - A string.
        prefix           - A string to be inserted before 'text'.
        indent           - The number of spaces to insert before 'text'.
        decoration       - Decorators for the text. (color/style options are listed as keys in
                           Console.colors)
        newline          - Boolean. If True, prints each string passed to Console.out on its own
                           line, sequentially.
        newline_at_start - Boolean. Set to True to keep the cursor at the end of the line instead
                           of sending it to the next line. This is useful if you need to frequently
                           erase/overwrite a line that has just been printed.
        """
        Console.queue.insert(0, {
            "text": text,
            "prefix": prefix,
            "indent": indent,
            "decoration": decoration,
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

            text, prefix, indent, decoration, newline, newline_at_start = ( last_item[key] for key in ("text","prefix","indent","decoration","newline","newline_at_start") )

            if newline and newline_at_start:
                write_txt("\n")
            if decoration is not None:
                if type(decoration) in map(type, [[],np.array([])]):
                    for item in decoration:
                        write_txt( Console.colors[item] )
                elif type(decoration) is type(""):
                    write_txt( Console.colors[decoration] )
            indentation = indent * "    "
            write_txt(indentation+str(prefix)+str(text))
            if decoration is not None:
                write_txt( Console.colors["RESET"] )
            if newline and not newline_at_start:
                write_txt("\n")

        Console.busy = False

    # Some terminal message presets for error logging.
    @classmethod
    def warn(Console, text, prefix="", indent=0):
        Console.out("WARNING: "+text, prefix=prefix, indent=indent, decoration="RED")
    @classmethod
    def note(Console, text, prefix="", indent=0):
        Console.out("NOTE: "+text, prefix=prefix, indent=indent, decoration="BOLD")
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

        def __init__(self, time_limit=np.inf, delay=None, decoration="CYAN", icon="ellipsis", text="", indent=0, attempt_number=1, print_attempts=False, no_spinner=False, no_print=False):
            self.no_print = no_print
            self.no_spinner = no_spinner
            self.attempt_number = attempt_number
            self.print_attempts = print_attempts
            self.indent = indent
            self.text = text
            self.time_limit = time_limit
            self.decoration = decoration
            self.spinner_generator = self.spinning_cursor(icon)
            if delay and float(delay): self.delay = delay

        def update_text(self, text):
            """
            Updates the text that precedes the spinner.
            """
            self.text = text
            if self.no_spinner and not self.no_print:
                Console.out(self.text+"...... "+str(round(self.Dt, 2))+"s", decoration=self.decoration, indent=self.indent)

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

                    sys.stdout.write( Console.colors[self.decoration] )
                    sys.stdout.write( spinner_frame )
                    sys.stdout.write( Console.colors["RESET"] )
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
                    Console.out(self.text, decoration="GREEN", indent=self.indent)
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
                    Console.out(text, decoration="GREEN", indent=indent)
                else:
                    Console.out(end_text+" in "+Dt+".", decoration="GREEN", indent=indent)

            if return_time:
                return ( output, Dt )
            else:
                return output
    return f

def save_table_to_txt( filename, table, column_titles=None, column_padding=3 ):

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

def print_dict(d, indent=1, ofe=False, top_level=True):
    """
    Prints dictionaries such that more nested data receive more indentation.
    """

    def quote_str(el):
        if type(el) is str:
            return "'"+el+"'"
        else:
            return str(el)

    if top_level:
        Console.out("{")
    keys = d.keys()
    len_keys = len(keys)
    for i, key in enumerate(keys):
        on_final_element = i == len_keys-1
        if type(d[key]) is not type({}):
            line =  f"{'   '*indent}{ quote_str(key) }: { quote_str(d[key]) }" + ( "" if on_final_element else "," )
            Console.out( line )
        else:
            Console.out( f"{'   '*indent}{ quote_str(key) }:" + " {" )
            print_dict( d[key], indent=indent+1, ofe=on_final_element, top_level=False )
    if top_level:
        Console.out("}")
    else:
        Console.out( ('   '*(indent-1))+"}" + ( "" if ofe else "," ) )

def cartprod(*arrays):
    """
    Generates the Cartesian product of any number of input arrays.
    The input arrays can contain elements of any data type.
    """
    for element in arrays:
        try:
            element[len(element)-1] # check that the elements are indexable
        except:
            raise ValueError

    arrays = arrays[::-1]
    output = [ [element] for element in arrays[0] ]
    for array in arrays[1:]:
        new_output = []
        for array_element in array:
            for output_array in output:
                new_output.append( [array_element] + output_array )
        del output
        output = new_output
    return output

def subset_colormap(original_cmap, new_min=0, new_max=1):
    """
    Creates a matplotlib colormap from a subset of an existing matplotlib colormap.

    original_cmap   - (colormap) The matplotlib colormap to be sampled from.
    new_min         - (float/int, ranging from 0-1) The 'start' of the subset.
    new_max         - (float/int, ranging from 0-1) The 'end' of the subset.

    output          - (colormap)
    """
    import matplotlib.colors as mcol
    cmap = plt.get_cmap(original_cmap)
    new_map = mcol.LinearSegmentedColormap.from_list(
            original_cmap + '_subset',
            cmap(np.linspace(new_min, new_max, 100))
        )
    return new_map

def standard_notation(number_str):
    """
    Converts a number, repsented as a string, from scientific notation to standard notation.

    number_str  - (string) A string representing a number.
    """
    assert type(number_str) is str
    if "e" in number_str:
        split_str = number_str.split("e")
        value_str = split_str[0].replace(".", "")
        exponent = int(split_str[1])
        if exponent >= 0:
            value_str = value_str[:exponent+1]
            if len(value_str[exponent+1:]) > 0:
                value_str += "." + value_str[exponent+1:]
        elif exponent <= -1:
            prefix = "0."
            value_str = prefix + "0"*(abs(exponent)-1) + value_str
        number_str = value_str
    return number_str

def gen_primes():
    """
    Sieve of Eratosthenes
    Code by David Eppstein, UC Irvine, 28 Feb 2002
    http://code.activestate.com/recipes/117119/

    Generate an infinite sequence of prime numbers.
    """

    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    D={}
    q=2 # The running integer that's checked for primeness
    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            yield q
            D[q*q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            for p in D[q]:
                D.setdefault(p+q,[]).append(p)
            del D[q]
        q += 1
