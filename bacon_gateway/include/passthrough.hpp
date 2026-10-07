#pragma once

#include <cstddef>
#include <cstdint>

// =====================================================================
//  passthrough.hpp
//
//  ロボマス開発ボードへ送るUARTフレームは
//      [コマンドID, 値上位, 値下位] の3バイト組を並べ、COBSで包んだもの
//  である（uart.cpp の UART::uart_send を参照）。
//
//  2026-08-07以降、自動走行中の足回り指令はegg8(Jetson)がこの形式のまま
//  組み立ててイーサネットで送ってくる。bacon6はそのバイト列を書き換えず
//  そのまま開発ボードへ流す（パススルー）。
//
//  ただし bacon6 は「どのコマンドIDが入っているか」だけは知る必要がある:
//   ・egg8が送っていないコマンド（ARM/GM/昇降/回収/GPIO など）は、従来通り
//     MU3プロポ由来の値で bacon6 が補完フレームとして送る。将来egg8が
//     それらも送るようになれば、自動的に bacon6 は送らなくなる。
//   ・減速停止・非常停止のときは中継をやめて自前の停止フレームを出すため、
//     直前に流した車輪値を知っている必要がある（ランプの連続性）。
//
//  ここでの解析は「読むだけ」で、中継するバイト列には一切触れない。
// =====================================================================
struct UartCommandFrame
{
    // UDPで運べるUARTバイト列の上限。COBS済み64バイト＝生62バイト＝20コマンド。
    static constexpr std::size_t max_encoded  = 64;
    static constexpr std::size_t max_commands = 21;

    std::size_t count = 0;
    uint8_t     command[max_commands] = {0};
    int16_t     value[max_commands]   = {0};

    // 指定コマンドIDが含まれていれば値を返す。重複IDは解析時に拒否する。
    bool find(uint8_t id, int16_t& out) const;
    bool contains(uint8_t id) const;
};

// COBS済みフレームを解析する。壊れていればfalseを返し、呼び出し側は
// 中継せずローカル生成のフレームへフォールバックすること。
//
// encoded は encode_cobs の出力そのもの（末尾の区切り0x00を含む）。
bool parse_uart_command_frame(const uint8_t* encoded, std::size_t length,
                              UartCommandFrame& out);
