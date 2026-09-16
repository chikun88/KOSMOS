import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class PowerControlError(RuntimeError):
    pass


@dataclass
class PowerControlConfig:
    enabled: bool = False
    gpio_pin: int = -1
    gpio_mode: str = 'BOARD'
    gpio_backend: str = 'auto'
    active_high: bool = True
    settle_sec: float = 0.2
    off_on_shutdown: bool = True


class _JetsonGpioBackend:
    def __init__(self, pin: int, mode: str, active_high: bool) -> None:
        try:
            import Jetson.GPIO as GPIO
        except ImportError as error:
            raise PowerControlError('Jetson.GPIO is not installed') from error

        mode_name = str(mode).upper()
        if not hasattr(GPIO, mode_name):
            raise PowerControlError(f'Unsupported Jetson.GPIO mode: {mode}')
        self.GPIO = GPIO
        self.pin = int(pin)
        self.active_value = GPIO.HIGH if active_high else GPIO.LOW
        self.inactive_value = GPIO.LOW if active_high else GPIO.HIGH
        GPIO.setmode(getattr(GPIO, mode_name))
        GPIO.setup(self.pin, GPIO.OUT, initial=self.inactive_value)

    def set_enabled(self, enabled: bool) -> None:
        self.GPIO.output(
            self.pin,
            self.active_value if enabled else self.inactive_value,
        )

    def close(self) -> None:
        self.GPIO.cleanup(self.pin)


class _SysfsGpioBackend:
    def __init__(self, pin: int, active_high: bool) -> None:
        self.pin = int(pin)
        self.active_value = '1' if active_high else '0'
        self.inactive_value = '0' if active_high else '1'
        self.path = Path('/sys/class/gpio') / f'gpio{self.pin}'
        if not self.path.exists():
            try:
                Path('/sys/class/gpio/export').write_text(str(self.pin))
            except OSError as error:
                raise PowerControlError(
                    f'Could not export sysfs GPIO {self.pin}'
                ) from error
        try:
            (self.path / 'direction').write_text('out')
            (self.path / 'value').write_text(self.inactive_value)
        except OSError as error:
            raise PowerControlError(
                f'Could not configure sysfs GPIO {self.pin}'
            ) from error

    def set_enabled(self, enabled: bool) -> None:
        try:
            (self.path / 'value').write_text(
                self.active_value if enabled else self.inactive_value
            )
        except OSError as error:
            raise PowerControlError(
                f'Could not write sysfs GPIO {self.pin}'
            ) from error

    def close(self) -> None:
        pass


class MeasurementWheelPowerSwitch:
    """Controls an external 5 V load switch for encoders and buffer ICs."""

    def __init__(self, config: PowerControlConfig) -> None:
        self.config = config
        self.backend: Optional[object] = None
        self.enabled = False
        if not config.enabled:
            return
        if int(config.gpio_pin) < 0:
            raise PowerControlError(
                'power_control.enabled is true, but gpio_pin is not set'
            )

        backend_name = str(config.gpio_backend).lower()
        if backend_name == 'auto':
            if str(config.gpio_mode).lower() == 'sysfs':
                self.backend = _SysfsGpioBackend(
                    config.gpio_pin,
                    config.active_high,
                )
            else:
                self.backend = _JetsonGpioBackend(
                    config.gpio_pin,
                    config.gpio_mode,
                    config.active_high,
                )
        elif backend_name == 'jetson_gpio':
            self.backend = _JetsonGpioBackend(
                config.gpio_pin,
                config.gpio_mode,
                config.active_high,
            )
        elif backend_name == 'sysfs':
            self.backend = _SysfsGpioBackend(
                config.gpio_pin,
                config.active_high,
            )
        else:
            raise PowerControlError(
                f'Unsupported power GPIO backend: {config.gpio_backend}'
            )

    def turn_on(self) -> None:
        if self.backend is None:
            return
        self.backend.set_enabled(True)
        self.enabled = True
        time.sleep(max(0.0, float(self.config.settle_sec)))

    def turn_off(self) -> None:
        if self.backend is None:
            return
        self.backend.set_enabled(False)
        self.enabled = False

    def close(self) -> None:
        if self.backend is None:
            return
        if self.config.off_on_shutdown:
            self.turn_off()
        self.backend.close()
