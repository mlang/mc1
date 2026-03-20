import ast
import asyncio
import code
import concurrent.futures
import contextvars
import inspect
import readline
import threading
import types


class AsyncInteractiveConsole(code.InteractiveConsole):
    def __init__(self, locals=None, *, loop: asyncio.AbstractEventLoop):
        super().__init__(locals=locals)
        # Make the compiler accept top-level 'await'
        self.compile.compiler.flags |= ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

        self.loop = loop
        self.context = contextvars.copy_context()

    def runcode(self, code_obj):
        """
        code_obj is a compiled code object produced by InteractiveConsole.
        For top-level await, it behaves like a function body and may return a coroutine.
        """
        fut = concurrent.futures.Future()

        def run_in_loop():
            try:
                func = types.FunctionType(code_obj, self.locals)
                result = func()  # either normal result or coroutine (for top-level await)
            except SystemExit:
                raise
            except BaseException as e:
                fut.set_exception(e)
                return

            if inspect.iscoroutine(result):
                task = self.loop.create_task(result, context=self.context)
                task.add_done_callback(
                    lambda t: fut.set_exception(t.exception())
                    if t.exception() else fut.set_result(t.result())
                )
            else:
                fut.set_result(result)

        # Schedule evaluation onto the event loop thread
        self.loop.call_soon_threadsafe(run_in_loop, context=self.context)

        try:
            fut.result()  # block REPL thread; loop thread keeps running
        except SystemExit:
            raise
        except BaseException:
            self.showtraceback()


def interact(locals=None, init=None):
    loop = asyncio.new_event_loop()
    def loop_thread_main():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=loop_thread_main, name="asyncio-loop", daemon=True)
    t.start()

    if init is not None:
        asyncio.run_coroutine_threadsafe(init(locals), loop).result()

    console = AsyncInteractiveConsole(locals=locals, loop=loop)
    try:
        console.interact(banner='Async REPL (top-level await enabled). Ctrl-D to exit.')
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=1)

from mc1.clock import *

async def init(locals):
    locals['clock'].start()

if __name__ == "__main__":
    interact({"asyncio": asyncio, "clock": LogicalClock()}, init=init)
