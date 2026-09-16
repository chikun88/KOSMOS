import ctypes
import ctypes.util
from typing import List, Optional, Sequence


class ContecCounterError(RuntimeError):
    pass


class ContecCounter:
    """Small ctypes wrapper for the Contec CNT Linux C API."""

    CNT_SIGTYPE_ISOLATE = 0
    CNT_SIGTYPE_TTL = 1
    CNT_SIGTYPE_LINERECEIVER = 2
    CNT_DIR_DOWN = 0
    CNT_DIR_UP = 1
    CNT_MODE_1PHASE = 0
    CNT_MODE_2PHASE = 1
    CNT_MODE_GATECONTROL = 2
    CNT_MUL_X1 = 0
    CNT_MUL_X2 = 1
    CNT_MUL_X4 = 2
    CNT_CLR_ASYNC = 0
    CNT_CLR_SYNC = 1
    CNT_ZPHASE_NOT_USE = 1
    CNT_ZPHASE_NEXT_ONE = 2
    CNT_ZPHASE_EVERY_TIME = 3
    CNT_ZLOGIC_POSITIVE = 0
    CNT_ZLOGIC_NEGATIVE = 1

    def __init__(
        self,
        *,
        device_name: str,
        channels: Sequence[int],
        library_path: str = '',
    ) -> None:
        self.device_name = str(device_name).encode('ascii')
        self.channels = [int(channel) for channel in channels]
        if not self.channels:
            raise ValueError('at least one CNT channel is required')
        self.library_path = str(library_path)
        self.library = self._load_library()
        self._configure_functions()
        self.device_id: Optional[int] = None
        self.started = False

    def _load_library(self):
        candidates = []
        if self.library_path:
            candidates.append(self.library_path)
        found = ctypes.util.find_library('ccnt')
        if found:
            candidates.append(found)
        candidates.extend(['libccnt.so', 'ccnt'])

        errors = []
        for candidate in candidates:
            try:
                return ctypes.CDLL(candidate)
            except OSError as error:
                errors.append(f'{candidate}: {error}')
        raise ContecCounterError(
            'Could not load Contec CNT library. Install the CNT-3204IN-USB '
            'Linux driver or set cnt_library. Tried: ' + '; '.join(errors)
        )

    def _configure_functions(self) -> None:
        short_p = ctypes.POINTER(ctypes.c_short)
        ulong_p = ctypes.POINTER(ctypes.c_ulong)

        self.library.CntInit.argtypes = [ctypes.c_char_p, short_p]
        self.library.CntInit.restype = ctypes.c_long
        self.library.CntSetZMode.argtypes = [
            ctypes.c_short,
            ctypes.c_short,
            ctypes.c_short,
        ]
        self.library.CntSetZMode.restype = ctypes.c_long
        self.library.CntSetZLogic.argtypes = [
            ctypes.c_short,
            ctypes.c_short,
            ctypes.c_short,
        ]
        self.library.CntSetZLogic.restype = ctypes.c_long
        self.library.CntSelectChannelSignal.argtypes = [
            ctypes.c_short,
            ctypes.c_short,
            ctypes.c_short,
        ]
        self.library.CntSelectChannelSignal.restype = ctypes.c_long
        self.library.CntSetCountDirection.argtypes = [
            ctypes.c_short,
            ctypes.c_short,
            ctypes.c_short,
        ]
        self.library.CntSetCountDirection.restype = ctypes.c_long
        self.library.CntSetOperationMode.argtypes = [
            ctypes.c_short,
            ctypes.c_short,
            ctypes.c_short,
            ctypes.c_short,
            ctypes.c_short,
        ]
        self.library.CntSetOperationMode.restype = ctypes.c_long
        self.library.CntSetDigitalFilter.argtypes = [
            ctypes.c_short,
            ctypes.c_short,
            ctypes.c_short,
        ]
        self.library.CntSetDigitalFilter.restype = ctypes.c_long
        self.library.CntStartCount.argtypes = [
            ctypes.c_short,
            short_p,
            ctypes.c_short,
        ]
        self.library.CntStartCount.restype = ctypes.c_long
        self.library.CntReadCount.argtypes = [
            ctypes.c_short,
            short_p,
            ctypes.c_short,
            ulong_p,
        ]
        self.library.CntReadCount.restype = ctypes.c_long
        self.library.CntStopCount.argtypes = [
            ctypes.c_short,
            short_p,
            ctypes.c_short,
        ]
        self.library.CntStopCount.restype = ctypes.c_long
        self.library.CntExit.argtypes = [ctypes.c_short]
        self.library.CntExit.restype = ctypes.c_long

    def _channel_array(self):
        array_type = ctypes.c_short * len(self.channels)
        return array_type(*self.channels)

    def open(self) -> None:
        device_id = ctypes.c_short()
        ret = self.library.CntInit(self.device_name, ctypes.byref(device_id))
        if ret != 0:
            raise ContecCounterError(
                f'CntInit failed for {self.device_name.decode()}: {ret}'
            )
        self.device_id = int(device_id.value)

    def configure_channels(
        self,
        *,
        signal_type: int = CNT_SIGTYPE_ISOLATE,
        count_direction: int = CNT_DIR_UP,
        operation_phase: int = CNT_MODE_2PHASE,
        multiplier: int = CNT_MUL_X1,
        sync_clear: int = CNT_CLR_ASYNC,
        z_phase: int = CNT_ZPHASE_NOT_USE,
        z_logic: int = CNT_ZLOGIC_POSITIVE,
        digital_filter: int = 0,
    ) -> None:
        if self.device_id is None:
            raise ContecCounterError('configure_channels called before CntInit')
        for channel in self.channels:
            self._call_channel(
                'CntSetZMode',
                channel,
                z_phase,
            )
            self._call_channel(
                'CntSetZLogic',
                channel,
                z_logic,
            )
            self._call_channel(
                'CntSelectChannelSignal',
                channel,
                signal_type,
            )
            self._call_channel(
                'CntSetCountDirection',
                channel,
                count_direction,
            )
            ret = self.library.CntSetOperationMode(
                ctypes.c_short(self.device_id),
                ctypes.c_short(channel),
                ctypes.c_short(operation_phase),
                ctypes.c_short(multiplier),
                ctypes.c_short(sync_clear),
            )
            if ret != 0:
                raise ContecCounterError(
                    f'CntSetOperationMode failed for channel {channel}: {ret}'
                )
            self._call_channel(
                'CntSetDigitalFilter',
                channel,
                digital_filter,
            )

    def _call_channel(self, function_name: str, channel: int, value: int) -> None:
        if self.device_id is None:
            raise ContecCounterError(f'{function_name} called before CntInit')
        function = getattr(self.library, function_name)
        ret = function(
            ctypes.c_short(self.device_id),
            ctypes.c_short(channel),
            ctypes.c_short(value),
        )
        if ret != 0:
            raise ContecCounterError(
                f'{function_name} failed for channel {channel}: {ret}'
            )

    def start(self) -> None:
        if self.device_id is None:
            raise ContecCounterError('CntStartCount called before CntInit')
        ret = self.library.CntStartCount(
            ctypes.c_short(self.device_id),
            self._channel_array(),
            ctypes.c_short(len(self.channels)),
        )
        if ret != 0:
            raise ContecCounterError(f'CntStartCount failed: {ret}')
        self.started = True

    def read(self) -> List[int]:
        if self.device_id is None:
            raise ContecCounterError('CntReadCount called before CntInit')
        data_type = ctypes.c_ulong * len(self.channels)
        data = data_type()
        ret = self.library.CntReadCount(
            ctypes.c_short(self.device_id),
            self._channel_array(),
            ctypes.c_short(len(self.channels)),
            data,
        )
        if ret != 0:
            raise ContecCounterError(f'CntReadCount failed: {ret}')
        return [int(value) for value in data]

    def close(self) -> None:
        if self.device_id is None:
            return
        if self.started:
            self.library.CntStopCount(
                ctypes.c_short(self.device_id),
                self._channel_array(),
                ctypes.c_short(len(self.channels)),
            )
            self.started = False
        self.library.CntExit(ctypes.c_short(self.device_id))
        self.device_id = None
