import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import traceback
import os
import random
import time
import dtw
from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_file
from collections import deque, Counter
import io

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_default_fallback_secret_key')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

DATA_FILENAME = "makemeahanzi/graphics.txt"
POEMS_SOURCES = {
    'poems.json': '唐詩三百首',
    'easypoems.json': '常見唐詩(較簡單)'
}
DEFAULT_POEMS_SOURCE = 'poems.json'
PLOTS_OUTPUT_DIR = "plots"

ALL_CHARACTERS_DATA = {}
VALID_POEM_LINES_MAP = {}
POEM_INFO_MAP_MAP = {}
GAME_LOAD_ERROR = None

RECENT_LINES_LIMIT = 10

def parse_svg_path(path_string, num_curve_points=30):
    points_list = []
    current_point = None
    subpath_start_point = None

    tokens = re.findall(r'[MLQCSZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?\s*', path_string)

    i = 0
    num_tokens = len(tokens)

    def get_coords(count):
        nonlocal i
        coords = [];
        if i + count > num_tokens:
            command_context = tokens[i-1].strip() if i > 0 else "START";
            remaining = num_tokens - i;
            raise ValueError(f"座標不足 (需要 {count}, 剩餘 {remaining}) 於指令 '{command_context}' 之後 (索引 {i}).");

        for k in range(count):
            coord_str = tokens[i].strip();
            coords.append(float(coord_str));
            i += 1;
        return coords;

    try:
        while i < num_tokens:
            command = tokens[i].strip();

            if not re.match(r'[MLQCSZz]', command):
                i += 1;
                continue;

            i += 1;

            if command == 'M':
                coords = get_coords(2);
                x, y = coords;
                current_point = np.array([x, y]);
                subpath_start_point = current_point.copy();
                points_list.append([current_point.tolist()]);

                while i + 1 < num_tokens and re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i].strip()) and re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+1].strip()):
                     coords = get_coords(2);
                     x, y = coords;
                     current_point = np.array([x, y]);
                     points_list[-1].append(current_point.tolist());

            elif command == 'L':
                 if current_point is None:
                      raise ValueError(f"L 指令於索引 {i-1} 需要先前的 M, L, Q, 或 C 指令.");
                 while True:
                    coords = get_coords(2);
                    x, y = coords;
                    current_point = np.array([x, y]);
                    points_list[-1].append(current_point.tolist());
                    if i + 1 >= num_tokens or not (re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i].strip()) and re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+1].strip())):
                        break;

            elif command == 'Q':
                 if current_point is None:
                      raise ValueError(f"Q 指令於索引 {i-1} 需要先前的 M, L, Q, 或 C 指令.");
                 while True:
                     coords = get_coords(4);
                     p0 = current_point;
                     p1 = np.array(coords[:2]);
                     p2 = np.array(coords[2:]);

                     curve_points = [];
                     if num_curve_points >= 2:
                         for t in np.linspace(0, 1, num_curve_points):
                             curve_points.append(((1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2).tolist());
                     elif num_curve_points == 1:
                          curve_points.append(p2.tolist());

                     if curve_points:
                         if points_list and points_list[-1] and (abs(points_list[-1][-1][0] - curve_points[0][0]) > 1e-6 or abs(points_list[-1][-1][1] - curve_points[0][1]) > 1e-6):
                              points_list[-1].extend(curve_points);
                         elif points_list:
                              points_list[-1].extend(curve_points[1:]);

                     current_point = p2;

                     if i + 3 >= num_tokens or not (
                        re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i].strip()) and
                        re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+1].strip()) and
                        re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+2].strip()) and
                        re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+3].strip())):
                         break;

            elif command == 'C':
                 if current_point is None:
                      raise ValueError(f"C 指令於索引 {i-1} 需要先前的 M, L, Q, 或 C 指令.");
                 while True:
                    coords = get_coords(6);
                    p0 = current_point;
                    p1 = np.array(coords[:2]);
                    p2_ctl = np.array(coords[2:4]);
                    p3 = np.array(coords[4:]);

                    curve_points = [];
                    if num_curve_points >= 2:
                        for t in np.linspace(0, 1, num_curve_points):
                            curve_points.append(((1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2_ctl + t**3 * p3).tolist());
                    elif num_curve_points == 1:
                         curve_points.append(p3.tolist());

                    if curve_points:
                         if points_list and points_list[-1] and (abs(points_list[-1][-1][0] - curve_points[0][0]) > 1e-6 or abs(points_list[-1][-1][1] - curve_points[0][1]) > 1e-6):
                             points_list[-1].extend(curve_points);
                         elif points_list:
                            points_list[-1].extend(curve_points[1:]);


                    current_point = p3;

                    if i + 5 >= num_tokens or not (
                         re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i].strip()) and
                         re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+1].strip()) and
                         re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+2].strip()) and
                         re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+3].strip()) and
                         re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+4].strip()) and
                         re.match(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', tokens[i+5].strip())):
                         break;

            elif command.lower() == 'z':
                if current_point is not None and subpath_start_point is not None and points_list:
                    if points_list[-1] and (abs(current_point[0] - subpath_start_point[0]) > 1e-6 or abs(current_point[1] - subpath_start_point[1]) > 1e-6):
                         points_list[-1].append(subpath_start_point.tolist());
                    current_point = subpath_start_point.copy();
                subpath_start_point = None;


            else:
                pass;


    except Exception as e:
        print(f"解析 SVG 路徑時發生錯誤: {e}");
        traceback.print_exc();
        return [np.array(seg_list) for seg_list in points_list if seg_list and isinstance(seg_list, list)];

    parsed_segments = [np.array(segment_points_list) for segment_points_list in points_list if segment_points_list];
    return parsed_segments;


def calculate_stroke_centroid(path_string):
    try:
        segments = parse_svg_path(path_string, num_curve_points=5);
        all_points = np.concatenate([s for s in segments if isinstance(s, np.ndarray) and s.shape[0] > 0]);
        if all_points.shape[0] > 0:
            return np.mean(all_points, axis=0);
        else:
             first_point = None;
             for seg_list in parse_svg_path(path_string, num_curve_points=2):
                  if seg_list and len(seg_list) > 0:
                       first_point = np.array(seg_list[0]);
                       break;
             return first_point if first_point is not None else np.array([512.0, 512.0]);


    except Exception as e:
        return np.array([512.0, 512.0]);


def load_character_data(filepath):
    character_data = {};
    full_filepath = os.path.join(app.root_path, filepath);

    print(f"嘗試載入筆畫資料檔案於: {full_filepath}");

    try:
        with open(full_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip();
                if not line:
                    continue;
                try:
                    data = json.loads(line);
                    if isinstance(data, dict) and 'character' in data and 'strokes' in data:
                        char = data['character'];
                        strokes = data['strokes'];
                        if isinstance(char, str) and len(char) == 1 and \
                           isinstance(strokes, list) and all(isinstance(s, str) for s in strokes):
                             character_data[char] = strokes;
                        else:
                             print(f"警告: 略過檔案中的無效資料格式 (字元或筆畫格式錯誤) 於行 {line_num}: {line}");
                    else:
                         print(f"警告: 略過檔案中的無效行 (缺少鍵 'character' 或 'strokes' 或非字典) {line_num}: {line}");
                except json.JSONDecodeError as e:
                    print(f"警告: 解碼 JSON 錯誤於行 {line_num}: {line} - {e}");
                except Exception as e:
                     print(f"處理行 {line_num} 時發生未知錯誤: {e}");
    except FileNotFoundError:
        error_msg = f"錯誤: 筆畫資料檔案未找到於 {full_filepath}. 請確認檔案存在.";
        print(error_msg);
        return {}, error_msg;
    except Exception as e:
        error_msg = f"讀取筆畫資料檔案 '{full_filepath}' 時發生錯誤: {e}";
        print(error_msg);
        traceback.print_exc();
        return {}, error_msg;

    print(f"成功載入 {len(character_data)} 個漢字的筆畫資料.");
    return character_data, None;


def load_poems_from_source(filepath, char_data):
    valid_lines = [];
    poem_info_map = {};
    full_filepath = os.path.join(app.root_path, filepath);

    print(f"嘗試載入詩詞檔案於: {full_filepath}");

    try:
        with open(full_filepath, 'r', encoding='utf-8') as f:
            poems_data = json.load(f);

        if not isinstance(poems_data, list):
             error_msg = f"詩詞檔案格式錯誤 ('{filepath}'): 根元素不是列表.";
             print(error_msg);
             return [], {}, error_msg;

        for poem in poems_data:
            if not isinstance(poem, dict) or 'title' not in poem or 'content' not in poem or not isinstance(poem['content'], list):
                 print(f"警告: 略過無效格式的詩詞條目 ('{filepath}'): {poem.get('title', '未知標題')}");
                 continue;

            title = poem.get('title', '未知標題');
            content = poem['content'];

            for line in content:
                if isinstance(line, str) and len(line) == 5:
                    is_valid_line = True;
                    for char in line:
                        if char not in char_data:
                            is_valid_line = False;
                            break;
                    if is_valid_line:
                        valid_lines.append(line);
                        poem_info_map[line] = {'title': title, 'content': [str(l) for l in content if isinstance(l, str)]};
                    else:
                        pass;

        valid_lines = list(set(valid_lines));
        print(f"成功載入並驗證 {len(valid_lines)} 條有效 (五言且字元存在) 詩句 來自 '{filepath}'.");
        if not valid_lines:
             return [], {}, f"沒有載入到任何有效的五言詩句 來自 '{filepath}'.";
        return valid_lines, poem_info_map, None;

    except FileNotFoundError:
        error_msg = f"錯誤: 詩詞檔案未找到於 {full_filepath}. 請確認檔案存在.";
        print(error_msg);
        return [], {}, error_msg;
    except json.JSONDecodeError as e:
        error_msg = f"錯誤: 解碼詩詞 JSON 檔案 '{full_filepath}' 錯誤: {e}";
        print(error_msg);
        return [], {}, error_msg;
    except Exception as e:
        error_msg = f"讀取詩詞檔案 '{full_filepath}' 時發生錯誤: {e}";
        print(error_msg);
        traceback.print_exc();
        return [], {}, error_msg;


ALL_CHARACTERS_DATA, char_load_error = load_character_data(DATA_FILENAME);
GLOBAL_POEMS_LOAD_ERRORS = {}
for source_key, source_name in POEMS_SOURCES.items():
     VALID_POEM_LINES, POEM_INFO_MAP, poems_load_error = load_poems_from_source(source_key, ALL_CHARACTERS_DATA)
     VALID_POEM_LINES_MAP[source_key] = VALID_POEM_LINES
     POEM_INFO_MAP_MAP[source_key] = POEM_INFO_MAP
     if poems_load_error:
         GLOBAL_POEMS_LOAD_ERRORS[source_key] = poems_load_error


if char_load_error:
    GAME_LOAD_ERROR = char_load_error
elif not VALID_POEM_LINES_MAP or all(not lines for lines in VALID_POEM_LINES_MAP.values()):
    GAME_LOAD_ERROR = "沒有載入到任何來源的有效五言詩句，無法開始遊戲。"
else:
    print("應用程式啟動資料載入完成.");


def preprocess_strokes(stroke_segments_list):
    processed_strokes = [];
    if not isinstance(stroke_segments_list, list):
        return [];

    for segment in stroke_segments_list:
        if isinstance(segment, np.ndarray) and segment.shape[0] > 0:
            center = np.mean(segment, axis=0);
            translated_segment = segment - center;
            processed_strokes.append(translated_segment);

    return processed_strokes;

def get_comparable_segments_with_original_index(char, char_data):
    segments_with_original_index = [];
    stroke_paths = char_data.get(char, []);
    if not stroke_paths:
         return [];

    for original_stroke_index, path_str in enumerate(stroke_paths):
         try:
             segments = parse_svg_path(path_str, num_curve_points=30);
             processed_segs = [seg for seg in preprocess_strokes(segments) if isinstance(seg, np.ndarray) and seg.shape[0] >= 2];
             segments_with_original_index.extend([(seg, original_stroke_index) for seg in processed_segs]);
         except Exception as e:
              print(f"Warning: 處理字元 '{char}' 筆畫 {original_stroke_index+1} 的筆畫路徑以準備 DTW 時發生錯誤: {e}");
              pass;

    return segments_with_original_index;

def plot_character_colored_by_history(target_char, char_history, thresholds, position_index, output_dir_relative):
    if target_char not in ALL_CHARACTERS_DATA:
        print(f"繪圖錯誤: 目標字元 '{target_char}' 不在筆畫資料中.");
        return None;

    target_stroke_paths = ALL_CHARACTERS_DATA[target_char];
    target_stroke_histories = char_history.get('stroke_histories', []);

    if not target_stroke_paths or len(target_stroke_paths) != len(target_stroke_histories):
         print(f"繪圖錯誤: 目標字元 '{target_char}' ({position_index}) 的筆畫資料 ({len(target_stroke_paths)}) 或歷史記錄 ({len(target_stroke_histories)}) 不一致.");
         return None;

    full_output_dir = os.path.join(app.static_folder, output_dir_relative);
    if not os.path.exists(full_output_dir):
        try:
            os.makedirs(full_output_dir);
        except OSError as e:
            print(f"錯誤: 無法建立輸出目錄 '{full_output_dir}': {e}. 無法儲存圖片.");
            return None;

    fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=100);
    ax.axis('off');
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0);

    ax.invert_yaxis();
    ax.set_aspect('equal', adjustable='box');

    padding = 200;
    ax.set_xlim(0 - padding, 1024 + padding);
    ax.set_ylim(0 - padding, 1024 + padding);


    threshold1 = thresholds.get('thresh1', 10000);
    threshold2 = thresholds.get('thresh2', 25000);

    color_pure_black = 0.0;
    color_light_gray = 0.9;
    color_dark_gray = 0.7;

    target_original_centroids = [calculate_stroke_centroid(path_str) for path_str in target_stroke_paths];

    for target_stroke_index, hist_info in enumerate(target_stroke_histories):
        historical_min_dist = hist_info.get('min_dist', float('inf'));
        historical_guess_char = hist_info.get('best_guess_char');
        historical_guess_stroke_index = hist_info.get('best_guess_stroke_index');

        display_stroke = (not np.isinf(historical_min_dist) and historical_min_dist < threshold2);

        if display_stroke:
            plot_path_str = None;
            historical_guess_stroke_paths = None;

            if historical_guess_char is not None and historical_guess_char in ALL_CHARACTERS_DATA:
                 historical_guess_stroke_paths = ALL_CHARACTERS_DATA[historical_guess_char];
                 if historical_guess_stroke_paths and historical_guess_stroke_index is not None and historical_guess_stroke_index < len(historical_guess_stroke_paths):
                    plot_path_str = historical_guess_stroke_paths[historical_guess_stroke_index];


            if plot_path_str:
                color_val = None;
                if historical_min_dist < threshold1:
                     color_val = color_pure_black;
                elif historical_min_dist >= threshold1 and historical_min_dist < threshold2:
                    dist_range = threshold2 - threshold1;
                    if dist_range > 0:
                        normalized_dist = (historical_min_dist - threshold1) / dist_range;
                        color_val = color_light_gray + normalized_dist * (color_dark_gray - color_light_gray);
                    else:
                        color_val = color_light_gray;

                if color_val is not None:
                    color_str = str(color_val);
                    try:
                        target_centroid = target_original_centroids[target_stroke_index];
                        historical_guess_centroid = calculate_stroke_centroid(plot_path_str);

                        translation = np.array([0.0, 0.0]);
                        if target_centroid is not None and historical_guess_centroid is not None:
                            translation = target_centroid - historical_guess_centroid;

                        hist_guess_segments = parse_svg_path(plot_path_str, num_curve_points=30);
                        for segment in hist_guess_segments:
                             if isinstance(segment, np.ndarray) and segment.shape[0] > 0:
                                  translated_segment = segment + translation;


                                  if translated_segment.shape[0] > 1:
                                      ax.plot(translated_segment[:, 0], translated_segment[:, 1], linestyle='-', linewidth=2, color=color_str, solid_capstyle='round', solid_joinstyle='round', zorder=2);
                                  elif translated_segment.shape[0] == 1:
                                      ax.plot(translated_segment[:, 0], translated_segment[:, 1], marker='o', markersize=10/2, color=color_str, zorder=2);

                    except Exception as e:
                         print(f"警告: 解析或繪製位置 {position_index} 目標筆畫 {target_stroke_index+1} 的歷史猜測筆畫時發生錯誤: {e}");
            else:
                 pass;

    filename = f"plot_{position_index}_{ord(target_char):04X}_colored_{int(time.time())}_{random.randint(0, 9999)}.png";
    image_relative_path = os.path.join(output_dir_relative, filename);
    image_full_path = os.path.join(app.static_folder, image_relative_path);

    try:
        plt.tight_layout(pad=0);
        fig.savefig(image_full_path, dpi=200, bbox_inches='tight', pad_inches=0);
        return image_relative_path;
    except Exception as e:
        print(f"錯誤: 儲存位置 {position_index} 目標 '{target_char}' 的著色圖片時發生錯誤: {e}");
        traceback.print_exc();
        return None;
    finally:
        plt.close(fig);

def initialize_game_session():
    global VALID_POEM_LINES_MAP, POEM_INFO_MAP_MAP, ALL_CHARACTERS_DATA;

    current_source = session.get('current_poem_source', DEFAULT_POEMS_SOURCE)
    valid_lines = VALID_POEM_LINES_MAP.get(current_source)
    poem_info_map = POEM_INFO_MAP_MAP.get(current_source)


    if not valid_lines:
        error_msg = f"當前選擇的題庫 ('{POEMS_SOURCES.get(current_source, current_source)}') 沒有可用的詩句，無法初始化遊戲.";
        print(error_msg);
        if current_source != DEFAULT_POEMS_SOURCE:
            print(f"嘗試回退到預設題庫 '{POEMS_SOURCES.get(DEFAULT_POEMS_SOURCE, DEFAULT_POEMS_SOURCE)}'.");
            current_source = DEFAULT_POEMS_SOURCE
            valid_lines = VALID_POEM_LINES_MAP.get(current_source)
            poem_info_map = POEM_INFO_MAP_MAP.get(current_source)
            if not valid_lines:
                 error_msg = f"預設題庫 ('{POEMS_SOURCES.get(DEFAULT_POEMS_SOURCE, DEFAULT_POEMS_SOURCE)}') 也沒有可用的詩句，無法初始化遊戲.";
                 print(error_msg);
                 session['current_poem_source'] = current_source
                 session.modified = True;
                 return False;
            else:
                 print(f"已回退到預設題庫.");


        else:
             session['current_poem_source'] = current_source
             session.modified = True;
             return False;

    session['current_poem_source'] = current_source
    print(f"使用題庫: '{POEMS_SOURCES.get(current_source, current_source)}'");


    current_state = session.get('poem_game_state', {})
    recent_lines_list = current_state.get('recent_lines', []);
    guess_count_history = current_state.get('guess_count_history', []);

    recent_lines = deque(recent_lines_list, maxlen=RECENT_LINES_LIMIT);

    available_lines = [line for line in valid_lines if line not in recent_lines];

    if not available_lines:
         print("沒有新的詩句可用 (所有詩句都在最近列表中)，重用最近的詩句.");
         target_line = random.choice(valid_lines);
    else:
         target_line = random.choice(available_lines);

    target_poem_info = poem_info_map.get(target_line, {'title': '未知詩名', 'content': [target_line]});

    recent_lines.append(target_line);

    char_histories = [];
    for char in target_line:
        stroke_paths = ALL_CHARACTERS_DATA.get(char, []);
        stroke_histories = [];
        for _ in stroke_paths:
            stroke_histories.append({
                'min_dist': float('inf'),
                'best_guess_char': None,
                'best_guess_stroke_index': None
            });
        char_histories.append({
            'target_char': char,
            'stroke_histories': stroke_histories
        });

    session['poem_game_state'] = {
        'target_line': target_line,
        'target_poem_info': target_poem_info,
        'char_histories': char_histories,
        'recent_lines': list(recent_lines),
        'guess_count': 0,
        'guess_count_history': guess_count_history
    };

    print(f"遊戲已初始化，目標詩句: '{target_line}'");
    session.modified = True;
    return True;

def validate_session_state():
    state = session.get('poem_game_state');
    if not isinstance(state, dict):
        return False, "Session 遊戲狀態遺失或格式無效.";

    required_keys = ['target_line', 'target_poem_info', 'char_histories', 'recent_lines', 'guess_count', 'guess_count_history'];
    if not all(key in state for key in required_keys):
        missing = [key for key in required_keys if key not in state]
        return False, f"Session 遊戲狀態不完整. 遺失鍵: {', '.join(missing)}.";

    target_line = state['target_line'];
    char_histories = state['char_histories'];
    poem_info = state['target_poem_info'];
    recent_lines = state['recent_lines'];
    guess_count = state['guess_count'];
    guess_count_history = state['guess_count_history']

    if not isinstance(target_line, str) or len(target_line) != 5:
        return False, "Session 目標詩句無效或長度錯誤.";

    if not isinstance(char_histories, list) or len(char_histories) != 5:
         return False, "Session 字元歷史記錄列表長度錯誤.";

    if not isinstance(poem_info, dict) or 'title' not in poem_info or 'content' not in poem_info or not isinstance(poem_info['content'], list):
         return False, "Session 詩詞資訊格式無效.";

    if not isinstance(recent_lines, list):
         return False, "Session 最近詩句記錄類型無效.";

    if not isinstance(guess_count, int) or guess_count < 0:
         return False, "Session 猜測計數無效.";

    if not isinstance(guess_count_history, list):
         return False, "Session 猜測歷史記錄類型無效.";


    for i in range(5):
         char_history = char_histories[i];
         if not isinstance(char_history, dict) or 'target_char' not in char_history or 'stroke_histories' not in char_history:
             return False, f"Session 位置 {i+1} 的字元歷史記錄結構無效.";

         if char_history['target_char'] != target_line[i]:
             print(f"Session char_histories[{i}] target_char mismatch: '{char_history['target_char']}' vs '{target_line[i]}'");
             return False, f"Session 位置 {i+1} 的目標字元 '{char_history['target_char']}' 不匹配當前詩句 '{target_line[i]}'.";

         if char_history['target_char'] not in ALL_CHARACTERS_DATA:
              print(f"Session char_histories[{i}] target_char '{char_history['target_char']}' not in ALL_CHARACTERS_DATA.");
              return False, f"Session 位置 {i+1} 的目標字元 '{char_history['target_char']}' 不在筆畫資料中.";

         target_char_stroke_count = len(ALL_CHARACTERS_DATA[char_history['target_char']]);
         if not isinstance(char_history['stroke_histories'], list) or len(char_history['stroke_histories']) != target_char_stroke_count:
              print(f"Session char_histories[{i}] stroke_histories invalid length: {len(char_history['stroke_histories'])} vs {target_char_stroke_count}");
              return False, f"Session 位置 {i+1} 的筆畫歷史記錄列表長度錯誤.";

         for j in range(len(char_history['stroke_histories'])):
              stroke_history = char_history['stroke_histories'][j];
              if not isinstance(stroke_history, dict) or 'min_dist' not in stroke_history or 'best_guess_char' not in stroke_history or 'best_guess_stroke_index' not in stroke_history:
                   return False, f"Session 位置 {i+1} 筆畫 {j+1} 的歷史記錄結構無效.";

    return True, "Session 狀態有效.";

def generate_stats_plot_buffer(guess_counts):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    if not guess_counts:
        ax.text(0.5, 0.5, '尚未完成任何詩詞猜測', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Total Guesses")
        ax.axis('off')
    else:
        count_map = Counter(guess_counts)
        sorted_counts = sorted(count_map.items())
        guess_numbers = [item[0] for item in sorted_counts]
        frequencies = [item[1] for item in sorted_counts]

        bars = ax.bar(guess_numbers, frequencies, color='#007bff')
        ax.set_xlabel("Guesses")
        ax.set_ylabel("times")
        ax.set_title("Total Guesses Graph")
        ax.set_xticks(guess_numbers)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

@app.route('/')
def index():
    if GAME_LOAD_ERROR:
         return render_template('index.html', game_error=GAME_LOAD_ERROR);

    is_valid_session, session_message = validate_session_state();
    if not is_valid_session:
         print(f"無效 Session 狀態: {session_message}. 嘗試重新初始化 Session.");
         if 'poem_game_state' in session:
             old_history = session['poem_game_state'].get('guess_count_history', [])
             session['poem_game_state'] = {'guess_count_history': old_history}
             session.modified = True
         if not initialize_game_session():
              error_msg = "無法初始化新遊戲，沒有可用的詩句.";
              print(error_msg);
              return render_template('index.html', game_error=error_msg);

    game_state = session.get('poem_game_state');
    if 'thresholds' not in session:
         session['thresholds'] = {'thresh1': 10000, 'thresh2': 25000};
         session.modified = True;
    thresholds = session['thresholds'];

    initial_state_for_js = {
        'target_line': game_state['target_line'],
        'is_correct_guess': False,
        'image_urls': [],
        'image_errors': [False, False, False, False, False],
        'poem_info': None,
        'messages': [{'type': 'warning', 'text': session_message}] if not is_valid_session else [],
        'game_ready': True,
        'thresholds': thresholds,
        'guess_count': game_state['guess_count'],
        'poem_sources': POEMS_SOURCES,
        'current_poem_source': session.get('current_poem_source', DEFAULT_POEMS_SOURCE),
        'guess_count_history': game_state.get('guess_count_history', [])
    };

    return render_template(
        'index.html',
        game_error=None,
        defaults=thresholds,
        initial_state=initial_state_for_js,
        poem_sources=POEMS_SOURCES,
        current_poem_source=session.get('current_poem_source', DEFAULT_POEMS_SOURCE)
    );


@app.route('/compare', methods=['POST'])
@app.route('/compare', methods=['POST'])
def compare_poem_line():
    response_data = {
        'status': 'success',
        'messages': [],
        'image_urls': [None] * 5,
        'image_errors': [True] * 5,
        'is_correct_guess': False,
        'poem_info': None,
        'thresholds': {},
        'guess_count': 0,
        'debug_info': {},
        'target_line': None
    }

    if GAME_LOAD_ERROR:
        response_data['status'] = 'error'
        response_data['messages'].append({'type': 'error', 'text': f"應用程式載入錯誤: {GAME_LOAD_ERROR}"})
        return jsonify(response_data), 500

    guess_line = request.form.get('guess_line', '').strip()
    try:
        thresh1 = float(request.form.get('thresh1', 10000))
        thresh2 = float(request.form.get('thresh2', 25000))
        response_data['thresholds'] = {'thresh1': thresh1, 'thresh2': thresh2}

        if thresh1 < 0 or thresh2 < 0:
             raise ValueError("閾值必須為非負數.")
        if thresh1 > thresh2:
             raise ValueError("閾值 1 不能大於閾值 2.")
        session['thresholds'] = response_data['thresholds']
        session.modified = True
    except ValueError as e:
        response_data['status'] = 'error'
        response_data['messages'].append({'type': 'error', 'text': f"無效的閾值輸入: {e}"})
        return jsonify(response_data), 400

    selected_source = request.form.get('poem_source', DEFAULT_POEMS_SOURCE)
    if selected_source not in POEMS_SOURCES:
         selected_source = DEFAULT_POEMS_SOURCE
         response_data['messages'].append({'type': 'warning', 'text': '無效的題庫選擇，已回退到預設題庫。'})

    if session.get('current_poem_source', DEFAULT_POEMS_SOURCE) != selected_source:
         print(f"使用者變更了題庫來源從 '{session.get('current_poem_source')}' 到 '{selected_source}'. 初始化新遊戲.")
         session['current_poem_source'] = selected_source
         session.modified = True
         if not initialize_game_session(): # Make sure initialize_game_session is defined
              error_msg = f"無法初始化新遊戲，當前選擇的題庫 ('{POEMS_SOURCES.get(selected_source)}') 沒有可用的詩句."
              response_data['status'] = 'error'
              response_data['messages'].append({'type': 'error', 'text': error_msg})
              return jsonify(response_data), 500

    is_valid_session, session_message = validate_session_state() # Make sure validate_session_state is defined
    if not is_valid_session:
        response_data['status'] = 'error'
        response_data['messages'].append({'type': 'error', 'text': f"遊戲狀態錯誤或已過期: {session_message}. 請重新開始新遊戲."})
        print(f"Session 無效於 /compare, 嘗試重新初始化: {session_message}")
        if 'poem_game_state' in session:
            old_history = session['poem_game_state'].get('guess_count_history', [])
            session['poem_game_state'] = {'guess_count_history': old_history} # Preserve history if possible
            session.modified = True
        if not initialize_game_session():
             error_msg = f"無法初始化新遊戲，當前選擇的題庫 ('{POEMS_SOURCES.get(session.get('current_poem_source'))}') 沒有可用的詩句."
             response_data['status'] = 'error'
             response_data['messages'].append({'type': 'error', 'text': error_msg})
             return jsonify(response_data), 500

    if len(guess_line) != 5:
        response_data['status'] = 'error'
        response_data['messages'].append({'type': 'error', 'text': f"請輸入剛好五個漢字進行猜測 (你輸入了 {len(guess_line)} 個字)."})
        return jsonify(response_data), 400

    missing_chars = [char for char in guess_line if char not in ALL_CHARACTERS_DATA]
    if missing_chars:
        response_data['status'] = 'error'
        response_data['messages'].append({'type': 'error', 'text': f"你的猜測詩句中包含不合法的字元: {''.join(missing_chars)}."})
        return jsonify(response_data), 400

    game_state = session['poem_game_state']
    target_line = game_state['target_line']
    char_histories = game_state['char_histories']

    game_state['guess_count'] += 1
    response_data['guess_count'] = game_state['guess_count']
    response_data['target_line'] = target_line

    current_valid_lines = VALID_POEM_LINES_MAP.get(session.get('current_poem_source', DEFAULT_POEMS_SOURCE), [])
    current_poem_info_map = POEM_INFO_MAP_MAP.get(session.get('current_poem_source', DEFAULT_POEMS_SOURCE), {})

    if any(char not in ALL_CHARACTERS_DATA for char in target_line) or target_line not in current_valid_lines:
         response_data['status'] = 'error'
         response_data['messages'].append({'type': 'error', 'text': f"內部錯誤: 當前目標詩句 '{target_line}' 包含無效字元或不在當前題庫中. 請重新開始新遊戲."})
         if initialize_game_session: initialize_game_session() # Attempt to reset
         return jsonify(response_data), 500

    any_plot_failed = False
    partial_failure = False

    guess_char_stroke_paths_map = {char: ALL_CHARACTERS_DATA.get(char, []) for char in set(guess_line)}

    for i in range(5): # Loop for each of the 5 characters in the poem line
        target_char = target_line[i]
        guess_char = guess_line[i]
        char_history = char_histories[i]

        target_stroke_paths = ALL_CHARACTERS_DATA.get(target_char, [])
        guess_stroke_paths = guess_char_stroke_paths_map.get(guess_char, [])

        # IMPORTANT REMINDER: The performance of DTW heavily depends on the number of points
        # in these segments. Ensure `get_comparable_segments_with_original_index` calls
        # `parse_svg_path` with a reduced `num_curve_points` (e.g., 5 to 10 instead of 30)
        # to avoid timeouts. See the placeholder for get_comparable_segments_with_original_index above.
        target_segments_with_original_index = get_comparable_segments_with_original_index(target_char, ALL_CHARACTERS_DATA)
        guess_segments_with_original_index = get_comparable_segments_with_original_index(guess_char, ALL_CHARACTERS_DATA)

        current_best_match_info_for_char = {}
        # plot_successful = False # This was in original, seems unused for DTW part

        try: # Outer try for this character's comparison logic
            if not target_stroke_paths or not guess_stroke_paths:
                 msg = f"位置 {i+1} 字元 '{target_char}' 或猜測字 '{guess_char}' 沒有筆畫資料."
                 response_data['messages'].append({'type': 'error', 'text': msg})
                 # response_data['status'] = 'warning' # Keep status as is or change if critical
                 any_plot_failed = True # This will likely prevent image generation or show error for it
                 continue # Skip to the next character if essential data is missing

            if not target_segments_with_original_index or not guess_segments_with_original_index:
                msg = f"位置 {i+1} ({target_char} vs {guess_char}): 沒有足夠點數的可比較筆畫段 (需要 >=2 點). 無法計算相似度."
                response_data['messages'].append({'type': 'warning', 'text': msg})
                partial_failure = True
                # If no segments, DTW matrix calculation below will be skipped or might fail if not handled.
                # The 'else' block handles this.
                # 'pass' was here, meaning it would fall through to plotting, which might be okay if char_history is empty.
            
            # Only proceed with DTW if we have segments to compare
            if target_segments_with_original_index and guess_segments_with_original_index:
                num_guess_segments_comp = len(guess_segments_with_original_index)
                num_target_segments_comp = len(target_segments_with_original_index)
                segment_dtw_matrix = np.full((num_guess_segments_comp, num_target_segments_comp), np.inf)

                for seg_g_idx, (seg_g, original_g_idx) in enumerate(guess_segments_with_original_index):
                    for seg_t_idx, (seg_t, original_t_idx) in enumerate(target_segments_with_original_index):
                        
                        # --- START: Integrated DTW calculation and error handling ---
                        if not isinstance(seg_g, np.ndarray) or seg_g.size == 0 or \
                           not isinstance(seg_t, np.ndarray) or seg_t.size == 0:
                            print(f"Warning: 無效或空的筆劃段用於 DTW 於位置 {i} (猜測 '{guess_char}' 筆畫 {original_g_idx+1} vs 目標 '{target_char}' 筆畫 {original_t_idx+1}). 跳過.")
                            segment_dtw_matrix[seg_g_idx, seg_t_idx] = float('inf')
                            continue # Skip this specific pair of segments

                        try:
                            euclidean_dist_func = lambda a, b: np.linalg.norm(np.array(a) - np.array(b), ord=2)
                            alignment = dtw.dtw(seg_g, seg_t)

                            if hasattr(alignment, 'distance'):
                                segment_dtw_matrix[seg_g_idx, seg_t_idx] = alignment.distance
                            elif isinstance(alignment, tuple) and len(alignment) >= 1 and isinstance(alignment[0], (float, np.floating)):
                                print(f"INFO: dtw.dtw 返回了元組。使用第一個元素作為距離，位置 {i} (猜測 '{guess_char}' 筆畫 {original_g_idx+1} vs 目標 '{target_char}' 筆畫 {original_t_idx+1}).")
                                segment_dtw_matrix[seg_g_idx, seg_t_idx] = alignment[0]
                            else:
                                print(f"ERROR: dtw.dtw 返回了無法處理的非預期類型或結構: {type(alignment)} 於位置 {i} (猜測 '{guess_char}' 筆畫 {original_g_idx+1} vs 目標 '{target_char}' 筆畫 {original_t_idx+1}). Value: {str(alignment)[:200]}")
                                segment_dtw_matrix[seg_g_idx, seg_t_idx] = float('inf')
                        
                        except Exception as dtw_e:
                            print(f"Warning: DTW 計算異常于位置 {i} (猜測 '{guess_char}' 筆畫 {original_g_idx+1} 段 vs 目標 '{target_char}' 筆畫 {original_t_idx+1} 段): {dtw_e}")
                            segment_dtw_matrix[seg_g_idx, seg_t_idx] = float('inf')
                        # --- END: Integrated DTW calculation and error handling ---

                # This part finds the best matching guess stroke for each target stroke
                num_target_original_strokes = len(target_stroke_paths)
                num_guess_original_strokes = len(guess_stroke_paths)

                for target_original_index in range(num_target_original_strokes):
                    min_dist_for_this_target_stroke = float('inf')
                    best_matching_guess_original_index_for_this_target_stroke = None

                    target_comparable_indices_map = [
                        idx for idx, (seg, original_idx) in enumerate(target_segments_with_original_index)
                        if original_idx == target_original_index
                    ]

                    if not target_comparable_indices_map:
                        continue

                    for guess_original_index in range(num_guess_original_strokes):
                        guess_comparable_indices_map = [
                            idx for idx, (seg, original_idx) in enumerate(guess_segments_with_original_index)
                            if original_idx == guess_original_index
                        ]

                        if not guess_comparable_indices_map:
                             continue

                        try:
                            # Ensure indices are valid for segment_dtw_matrix
                            if not guess_comparable_indices_map or not target_comparable_indices_map:
                                submatrix_min_dist = float('inf')
                            else:
                                # Check bounds to prevent IndexError if matrix is smaller than expected
                                max_g_idx = max(guess_comparable_indices_map)
                                max_t_idx = max(target_comparable_indices_map)
                                if max_g_idx < segment_dtw_matrix.shape[0] and max_t_idx < segment_dtw_matrix.shape[1]:
                                    submatrix = segment_dtw_matrix[np.ix_(guess_comparable_indices_map, target_comparable_indices_map)]
                                    submatrix_min_dist = np.min(submatrix) if submatrix.size > 0 else float('inf')
                                else:
                                    print(f"Warning: Index out of bounds for submatrix at pos {i}, target_stroke {target_original_index}, guess_stroke {guess_original_index}")
                                    submatrix_min_dist = float('inf')


                            if submatrix_min_dist < min_dist_for_this_target_stroke:
                                 min_dist_for_this_target_stroke = submatrix_min_dist
                                 best_matching_guess_original_index_for_this_target_stroke = guess_original_index

                        except Exception as submatrix_e:
                             print(f"Warning: 尋找位置 {i} 目標筆畫 {target_original_index+1} 與猜測筆畫 {guess_original_index+1} 之間的最小距離時發生錯誤: {submatrix_e}")
                             pass # min_dist_for_this_target_stroke remains unchanged or inf

                    if not np.isinf(min_dist_for_this_target_stroke) and best_matching_guess_original_index_for_this_target_stroke is not None:
                         current_best_match_info_for_char[target_original_index] = {
                             'guess_stroke_index': best_matching_guess_original_index_for_this_target_stroke,
                             'distance': min_dist_for_this_target_stroke
                         }
                         # Update history if this guess is better for this target stroke
                         if target_original_index < len(char_history.get('stroke_histories', [])):
                              hist_stroke_info = char_history['stroke_histories'][target_original_index]
                              if min_dist_for_this_target_stroke < hist_stroke_info.get('min_dist', float('inf')):
                                  hist_stroke_info['min_dist'] = min_dist_for_this_target_stroke
                                  hist_stroke_info['best_guess_char'] = guess_char
                                  hist_stroke_info['best_guess_stroke_index'] = best_matching_guess_original_index_for_this_target_stroke
            # else: (if not target_segments_with_original_index or not guess_segments_with_original_index)
            #   No DTW calculation was performed, char_history for this char won't be updated with new distances.
            #   Plotting will use existing char_history.

        except Exception as comp_e: # Handles errors in the character comparison logic (outside DTW loops)
            print(f"錯誤: 處理位置 {i} 字元 '{target_char}' vs '{guess_char}' 的比較邏輯時發生錯誤: {comp_e}")
            traceback.print_exc()
            response_data['messages'].append({'type': 'error', 'text': f"位置 {i+1} 字元比較失敗: {comp_e}"})
            # response_data['status'] = 'warning' # Or 'error' if critical
            any_plot_failed = True # Mark that plotting for this char might be compromised


        # Plotting logic for the current character
        try:
             image_relative_path = plot_character_colored_by_history(
                 target_char,
                 char_history, # char_history now contains updated (or old) best guesses
                 response_data['thresholds'],
                 i,
                 PLOTS_OUTPUT_DIR # Ensure PLOTS_OUTPUT_DIR is defined
             )

             if image_relative_path:
                 # Make sure url_for and static endpoint are correctly configured
                 response_data['image_urls'][i] = url_for('static', filename=image_relative_path)
                 response_data['image_errors'][i] = False
                 # plot_successful = True # This was in original, seems unused here
             else:
                 response_data['messages'].append({'type': 'warning', 'text': f"位置 {i+1} ({target_char}): 圖片生成失敗."})
                 partial_failure = True
                 any_plot_failed = True
        except Exception as plot_e:
            print(f"錯誤: 繪製位置 {i} 的圖片時發生錯誤: {plot_e}")
            traceback.print_exc()
            response_data['messages'].append({'type': 'warning', 'text': f"位置 {i+1} ({target_char}): 圖片生成時發生意外錯誤: {plot_e}"})
            partial_failure = True
            any_plot_failed = True
        # End of loop for each character

    if any_plot_failed:
         if response_data['status'] == 'success': # Only downgrade if it was initially success
              response_data['status'] = 'warning'
         # Add a general message if not already present and status became warning
         if partial_failure and response_data['status'] == 'warning' and \
            not any(m['type'] == 'warning' and "部分字元" in m['text'] for m in response_data['messages']) and \
            not any(m['type'] == 'error' for m in response_data['messages']):
              response_data['messages'].insert(0, {'type': 'warning', 'text': "部分字元的圖片生成失敗或有警告."})


    response_data['is_correct_guess'] = (guess_line == target_line)

    if response_data['is_correct_guess']:
        response_data['messages'].insert(0, {'type': 'success', 'text': "恭喜你，猜對了整句詩!"})
        response_data['poem_info'] = current_poem_info_map.get(target_line)
        # response_data['status'] = 'success' # Already success unless downgraded by errors

        if 'guess_count_history' not in game_state:
            game_state['guess_count_history'] = []
        game_state['guess_count_history'].append(game_state['guess_count'])
        game_state['guess_count'] = 0 # Reset for the next poem

    session['poem_game_state'] = game_state
    session.modified = True

    return jsonify(response_data)

@app.route('/new_poem', methods=['GET'])
def new_poem():
    if GAME_LOAD_ERROR:
        return render_template('index.html', game_error=GAME_LOAD_ERROR);

    current_source = request.args.get('source', session.get('current_poem_source', DEFAULT_POEMS_SOURCE))

    if current_source not in POEMS_SOURCES:
        print(f"嘗試使用無效的來源 '{current_source}', 回退到預設 '{DEFAULT_POEMS_SOURCE}'.");
        current_source = DEFAULT_POEMS_SOURCE

    session['current_poem_source'] = current_source
    session.modified = True;

    if not initialize_game_session():
         error_msg = f"無法開始新遊戲，當前選擇的題庫 ('{POEMS_SOURCES.get(current_source)}') 沒有可用的詩句.";
         print(error_msg);
         return render_template('index.html', game_error=error_msg);

    return redirect(url_for('index'));


@app.route('/stats_plot')
def stats_plot():
    game_state = session.get('poem_game_state', {})
    guess_count_history = game_state.get('guess_count_history', [])

    try:
        buf = generate_stats_plot_buffer(guess_count_history)
        return send_file(buf, mimetype='image/png', as_attachment=False)
    except Exception as e:
        print(f"生成統計圖表時發生錯誤: {e}")
        traceback.print_exc()
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.text(0.5, 0.5, '無法生成圖表', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png', as_attachment=False), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000));
    app.run(debug=True, host='0.0.0.0', port=port);