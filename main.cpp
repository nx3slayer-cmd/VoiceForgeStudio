#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Shell32.lib")
#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Kernel32.lib")

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Dynamic NVML API Signatures (GPU/VRAM Pre-flight without CUDA SDK link)
// ---------------------------------------------------------------------------
typedef int (*nvmlReturn_t)();
typedef nvmlReturn_t (*pfn_nvmlInit_v2)();
typedef nvmlReturn_t (*pfn_nvmlShutdown)();
typedef nvmlReturn_t (*pfn_nvmlDeviceGetHandleByIndex_v2)(unsigned int, void**);
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} NVML_MEMORY_T;
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo)(void*, NVML_MEMORY_T*);

struct GpuSpecs {
    bool hasNvidia = false;
    unsigned long long totalVramMB = 0;
    unsigned long long freeVramMB = 0;
};

GpuSpecs QueryHardwareAcceleration() {
    GpuSpecs specs;
    HMODULE hCuda = LoadLibraryW(L"nvcuda.dll");
    if (!hCuda) return specs;
    FreeLibrary(hCuda);
    specs.hasNvidia = true;

    HMODULE hNvml = LoadLibraryW(L"nvml.dll");
    if (hNvml) {
        auto nvmlInit = (pfn_nvmlInit_v2)GetProcAddress(hNvml, "nvmlInit_v2");
        auto nvmlShutdown = (pfn_nvmlShutdown)GetProcAddress(hNvml, "nvmlShutdown");
        auto nvmlGetHandle = (pfn_nvmlDeviceGetHandleByIndex_v2)GetProcAddress(hNvml, "nvmlDeviceGetHandleByIndex_v2");
        auto nvmlGetMem = (pfn_nvmlDeviceGetMemoryInfo)GetProcAddress(hNvml, "nvmlDeviceGetMemoryInfo");

        if (nvmlInit && nvmlShutdown && nvmlGetHandle && nvmlGetMem) {
            if (nvmlInit() == 0) {
                void* deviceHandle = nullptr;
                if (nvmlGetHandle(0, &deviceHandle) == 0) {
                    NVML_MEMORY_T memInfo{};
                    if (nvmlGetMem(deviceHandle, &memInfo) == 0) {
                        specs.totalVramMB = memInfo.total / (1024 * 1024);
                        specs.freeVramMB = memInfo.free / (1024 * 1024);
                    }
                }
                nvmlShutdown();
            }
        }
        FreeLibrary(hNvml);
    }
    return specs;
}

unsigned long long QueryAvailableSystemRamMB() {
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&mem)) {
        return mem.ullAvailPhys / (1024 * 1024);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// TCP Socket Health Probe for FastAPI / Uvicorn Server Port 8080
// ---------------------------------------------------------------------------
bool WaitForLocalServerReady(unsigned short port, int maxTimeoutSeconds) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;

    auto startTime = std::chrono::steady_clock::now();
    bool isReady = false;

    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - startTime).count() < maxTimeoutSeconds) {
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock != INVALID_SOCKET) {
            u_long mode = 1;
            ioctlsocket(sock, FIONBIO, &mode);

            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

            connect(sock, (sockaddr*)&addr, sizeof(addr));

            fd_set writeSet;
            FD_ZERO(&writeSet);
            FD_SET(sock, &writeSet);
            timeval tv{ 0, 200000 }; // 200ms

            if (select(0, nullptr, &writeSet, nullptr, &tv) > 0) {
                isReady = true;
                closesocket(sock);
                break;
            }
            closesocket(sock);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    WSACleanup();
    return isReady;
}

// ---------------------------------------------------------------------------
// Main Process Supervisor
// ---------------------------------------------------------------------------
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
    // 1. Single-Instance Protection
    HANDLE hMutex = CreateMutexW(NULL, TRUE, L"Global\\VoiceForgeStudio_SingleInstance_Mutex");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        MessageBoxW(NULL, 
            L"VoiceForge Master Studio is already running in the background.\nOpening existing instance in your browser...", 
            L"VoiceForge Studio", MB_ICONINFORMATION | MB_OK);
        ShellExecuteW(NULL, L"open", L"http://localhost:8080", NULL, NULL, SW_SHOWNORMAL);
        return 0;
    }

    // 2. Real-Time High-Priority Thread Scheduling
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // 3. Resolve Application Base Directory
    wchar_t exeBuffer[MAX_PATH];
    GetModuleFileNameW(NULL, exeBuffer, MAX_PATH);
    fs::path baseDir = fs::path(exeBuffer).parent_path();
    SetCurrentDirectoryW(baseDir.c_str());

    // 4. Pre-flight Hardware Verification
    GpuSpecs gpu = QueryHardwareAcceleration();
    unsigned long long ramMB = QueryAvailableSystemRamMB();

    if (ramMB < 2048) {
        MessageBoxW(NULL, 
            L"Warning: Less than 2 GB of physical RAM is available.\nAI synthesis may run out of memory.", 
            L"VoiceForge Pre-flight Notice", MB_ICONWARNING | MB_OK);
    }

    // 5. Environment Variables for Isolated Portability
    fs::path runtimeDir = baseDir / "runtime";
    fs::path sitePackages = runtimeDir / "Lib" / "site-packages";
    fs::path playwrightBrowsers = runtimeDir / "playwright-browsers";

    SetEnvironmentVariableW(L"PYTHONUNBUFFERED", L"1");
    SetEnvironmentVariableW(L"PLAYWRIGHT_BROWSERS_PATH", playwrightBrowsers.c_str());
    SetEnvironmentVariableW(L"HF_HUB_DISABLE_SYMLINKS_WARNING", L"1");

    // 6. Windows Job Object (Child Process Lifecycle Management)
    HANDLE hJob = CreateJobObjectW(NULL, NULL);
    if (hJob) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
        jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK;
        SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
    }

    // 7. Locate Runtime Executable
    fs::path pythonExe = runtimeDir / "python.exe";
    if (!fs::exists(pythonExe)) {
        pythonExe = L"python.exe"; // Fallback to PATH if runtime not yet bundled
    }

    fs::path appPy = baseDir / "app.py";
    if (!fs::exists(appPy)) {
        MessageBoxW(NULL, L"Fatal Error: 'app.py' not found in the application directory.", L"Missing File", MB_ICONERROR | MB_OK);
        return 1;
    }

    // 8. Launch Python Backend Process Suspended
    std::wstring cmd = L"\"" + pythonExe.wstring() + L"\" \"" + appPy.wstring() + L"\"";

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    DWORD creationFlags = CREATE_NO_WINDOW | CREATE_SUSPENDED;

    BOOL procCreated = CreateProcessW(
        NULL,
        &cmd[0],
        NULL,
        NULL,
        FALSE,
        creationFlags,
        NULL,
        baseDir.c_str(),
        &si,
        &pi
    );

    if (!procCreated) {
        std::wstring err = L"Failed to start Python runtime. Windows Error: " + std::to_wstring(GetLastError());
        MessageBoxW(NULL, err.c_str(), L"Engine Startup Error", MB_ICONERROR | MB_OK);
        return 1;
    }

    // Bind Process to Job Object before starting execution
    if (hJob) {
        AssignProcessToJobObject(hJob, pi.hProcess);
    }
    ResumeThread(pi.hThread);

    // 9. Asynchronous Connection Check & Browser Auto-Launch
    std::thread([hProcess = pi.hProcess]() {
        if (WaitForLocalServerReady(8080, 30)) {
            ShellExecuteW(NULL, L"open", L"http://localhost:8080", NULL, NULL, SW_SHOWNORMAL);
        } else {
            DWORD exitCode = 0;
            if (GetExitCodeProcess(hProcess, &exitCode) && exitCode != STILL_ACTIVE) {
                MessageBoxW(NULL, 
                    L"VoiceForge server shut down immediately after launch.\nCheck engine dependencies or model weights.", 
                    L"Runtime Stopped", MB_ICONERROR | MB_OK);
            } else {
                ShellExecuteW(NULL, L"open", L"http://localhost:8080", NULL, NULL, SW_SHOWNORMAL);
            }
        }
    }).detach();

    // 10. Supervise until Process Exits
    WaitForSingleObject(pi.hProcess, INFINITE);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (hJob) CloseHandle(hJob);
    if (hMutex) {
        ReleaseMutex(hMutex);
        CloseHandle(hMutex);
    }

    return 0;
}
