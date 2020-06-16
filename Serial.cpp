#include "Serial.h"


Serial::Serial(const std::string& port_name) {

    this->port_name = port_name;
    bool status;
    auto DEFAULT_BAUDRATE = CBR_115200;
    auto DEFAULT_BYTESIZE = 8;
    auto DEFAULT_STOPBITS = ONESTOPBIT;
    auto DEFAULT_PARITY = NOPARITY;

    comm_handle = CreateFileA(port_name.c_str(),                        // port name
        GENERIC_READ | GENERIC_WRITE,     // Read/Write
        0,                             // No Sharing
        0,                             // No Security
        OPEN_EXISTING,                    // Open existing port only
        0,                             // Non Overlapped I/O
        0
    );

    if (comm_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(std::string("Could not open serial port ") + port_name);
    }
    std::cout << "Opened connection to " << port_name << "\n";

    dcb_params.DCBlength = sizeof(dcb_params);
    status = GetCommState(comm_handle, &dcb_params);
    if (!status) {
        throw std::runtime_error("Could not read comm state");
    }

    dcb_params.BaudRate = DEFAULT_BAUDRATE;
    dcb_params.ByteSize = DEFAULT_BYTESIZE;
    dcb_params.StopBits = DEFAULT_STOPBITS;
    dcb_params.Parity = DEFAULT_PARITY;

    if (!SetCommState(comm_handle, &dcb_params)) {
        throw std::runtime_error("Serial::Serial: could not set DCB");
    }

    // Times in ms

    timeouts.ReadIntervalTimeout = 10;// 10;
    timeouts.ReadTotalTimeoutConstant = 0;//0;
    timeouts.ReadTotalTimeoutMultiplier = 0;//0;

    timeouts.WriteTotalTimeoutConstant = 1;// 50;
    timeouts.WriteTotalTimeoutMultiplier = 1;// 10;

    QUERY_SLEEP_DURATION = 0;

    if (!SetCommTimeouts(comm_handle, &timeouts)) {
        throw std::runtime_error("Serial::write: could not set timeouts");
    }

    // Clear buffer
    if (!PurgeComm(comm_handle, PURGE_RXCLEAR)) {
        throw std::runtime_error("Serial::Serial: Could not clear buffer");
    }

    // Required because the first values seems to be wrong sometimes
    test_serial();
}

Serial::~Serial() {
    CloseHandle(comm_handle);
}

void Serial::read(std::string& answer) {

    char   temp_char;
    DWORD  number_of_bytes_read = 1;

    if (!SetCommMask(comm_handle, EV_RXCHAR)) {
        throw std::runtime_error("Serial::read: Could not set comm mask");
    }

    std::stringstream ss;
    while (number_of_bytes_read > 0) {
        if (!ReadFile(comm_handle, &temp_char, sizeof(temp_char), &number_of_bytes_read, nullptr)) {
            throw std::runtime_error(std::string("Serial::read: Could not read serial port ") + port_name);
        }
        ss << temp_char;
        if (temp_char == '\n' || temp_char == '\0') {
            break;
        }
    }
    answer = ss.str();
}

void Serial::write(const std::string& message) {

    DWORD       number_of_bytes_to_write = message.size();
    DWORD       number_of_bytes_written = 0;

    bool status = WriteFile(
        comm_handle,
        message.c_str(),
        number_of_bytes_to_write,
        &number_of_bytes_written,
        nullptr
    );
    if (!status) {
        throw std::runtime_error(std::string("Serial::write: Could not write to serial port: ") + port_name);
    }
}

void Serial::query(const std::string& request, std::string& answer, int delay = 50) {
    write(request);
    // Clear buffer
    if (!PurgeComm(comm_handle, PURGE_RXCLEAR)) {
        throw std::runtime_error("Serial::Serial: Could not clear buffer");
    }
    Sleep(delay);
    
    
    read(answer);
}

std::string Serial::query(const std::string& request, int delay = 50) {
    write(request);
    // Clear buffer
    if (!PurgeComm(comm_handle, PURGE_RXCLEAR)) {
        throw std::runtime_error("Serial::Serial: Could not clear buffer");
    }
    Sleep(delay);

    std::string answer;

    read(answer);

    return answer;
}

void Serial::test_serial() {
    std::string temp_str;
    for (size_t i = 0; i < 10; i++) {
        query("1", temp_str);
        // std::cout << temp_str << "\n";
        Sleep(50);
    }
}
