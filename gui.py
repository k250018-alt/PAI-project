import os
import sys
import random
import pygame

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "gin rummy"))

from play import Play

pygame.init()
pygame.font.init()
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
except Exception:
    pass

WIDTH, HEIGHT = 1150, 720
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gin Rummy Desktop AI")

# Classic Solitaire green-felt palette
BG_DARK = (0, 100, 0)
BG_LIGHT = (34, 139, 34)
BG_FELT = (15, 120, 15)
SLOT_BG = (5, 75, 5)
GOLD = (255, 215, 0)
WHITE = (255, 255, 255)
BTN_GREEN = (30, 130, 55)
BTN_DARK = (20, 80, 35)

FONT_TITLE = pygame.font.SysFont("Trebuchet MS", 52, bold=True)
FONT_LARGE = pygame.font.SysFont("Trebuchet MS", 28, bold=True)
FONT_MED = pygame.font.SysFont("Trebuchet MS", 20, bold=True)
FONT_SMALL = pygame.font.SysFont("Trebuchet MS", 15, bold=True)

CARD_THEMES = [
    ("Classic Royale", os.path.join(BASE_DIR, "CuteCards - asset pack 0.1update", "CuteCards - asset pack", "ClassicRoyale.png")),
    ("Cute Pixel", os.path.join(BASE_DIR, "CuteCards - asset pack 0.1update", "CuteCards - asset pack", "CuteCards.png")),
    ("Outlined Pixel", os.path.join(BASE_DIR, "CuteCards - asset pack 0.1update", "CuteCards - asset pack", "CuteCards_outline.png")),
    ("Traditional", os.path.join(BASE_DIR, "CuteCards - asset pack 0.1update", "CuteCards - asset pack", "updated-card-2026.png")),
]


_FELT_CACHE = {}

def draw_felt_background(surface, width, height):
    """Draws a subtle solitaire-style green felt background with diamond texture (cached)."""
    cache_key = (width, height)
    if cache_key not in _FELT_CACHE:
        bg = pygame.Surface((width, height))
        bg.fill(BG_DARK)
        # Subtle diamond felt pattern
        pattern_color = (10, 110, 10)
        spacing = 28
        for y in range(-spacing, height + spacing, spacing):
            for x in range(-spacing, width + spacing, spacing):
                offset = spacing // 2 if (y // spacing) % 2 else 0
                cx, cy = x + offset, y
                pts = [(cx, cy - 4), (cx + 4, cy), (cx, cy + 4), (cx - 4, cy)]
                pygame.draw.polygon(bg, pattern_color, pts)
        # Soft vignette: darker edges
        for edge_w in range(60):
            alpha = int(40 * (1 - edge_w / 60))
            edge_surf = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.rect(edge_surf, (0, 0, 0, alpha), (edge_w, edge_w, width - 2*edge_w, height - 2*edge_w), 1)
            bg.blit(edge_surf, (0, 0))
        _FELT_CACHE[cache_key] = bg
    surface.blit(_FELT_CACHE[cache_key], (0, 0))


class SoundManager:
    """Manages audio effects for draws, discards, knocks, gins, clicks, and background music."""
    def __init__(self):
        self.sounds = {}
        sound_files = {
            "draw": "assets/sounds/card_draw.wav",
            "discard": "assets/sounds/card_discard.wav",
            "knock": "assets/sounds/knock.wav",
            "gin": "assets/sounds/gin.wav",
            "click": "assets/sounds/button_click.wav",
        }
        for name, rel_path in sound_files.items():
            full_path = os.path.join(BASE_DIR, rel_path)
            if os.path.exists(full_path):
                try:
                    self.sounds[name] = pygame.mixer.Sound(full_path)
                except Exception:
                    pass

        # Background music
        bgm_path = os.path.join(BASE_DIR, "assets", "sounds", "bgm_chill_lofi.wav")
        self.bgm_playing = False
        if os.path.exists(bgm_path):
            try:
                pygame.mixer.music.load(bgm_path)
                pygame.mixer.music.set_volume(0.3)  # 30% volume so it doesn't overpower SFX
                pygame.mixer.music.play(-1)  # Loop forever
                self.bgm_playing = True
            except Exception:
                pass

    def play(self, name: str):
        snd = self.sounds.get(name)
        if snd:
            try:
                snd.play()
            except Exception:
                pass

    def toggle_music(self):
        """Toggle background music on/off."""
        if self.bgm_playing:
            pygame.mixer.music.pause()
            self.bgm_playing = False
        else:
            pygame.mixer.music.unpause()
            self.bgm_playing = True


class CardAnimation:
    """Smooth ease-out cubic flying card animation."""
    def __init__(self, card_code, start_pos, end_pos, duration_ms=220, is_face_up=True):
        self.card_code = card_code
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration_ms = duration_ms
        self.start_ticks = pygame.time.get_ticks()
        self.is_face_up = is_face_up
        self.finished = False

    def get_pos(self):
        elapsed = pygame.time.get_ticks() - self.start_ticks
        t = min(1.0, elapsed / self.duration_ms)
        ease_t = 1.0 - (1.0 - t) ** 3
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * ease_t
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * ease_t
        if t >= 1.0:
            self.finished = True
        return (int(x), int(y))


class ScreenShake:
    """Screen vibration/shake effect on Knock or Gin."""
    def __init__(self):
        self.duration_ms = 0
        self.intensity = 0
        self.start_ticks = 0

    def start(self, duration_ms=450, intensity=12):
        self.duration_ms = duration_ms
        self.intensity = intensity
        self.start_ticks = pygame.time.get_ticks()

    def get_offset(self):
        if self.duration_ms <= 0:
            return (0, 0)
        elapsed = pygame.time.get_ticks() - self.start_ticks
        if elapsed >= self.duration_ms:
            self.duration_ms = 0
            return (0, 0)
        decay = 1.0 - (elapsed / self.duration_ms)
        curr_int = int(self.intensity * decay)
        if curr_int <= 0:
            return (0, 0)
        return (random.randint(-curr_int, curr_int), random.randint(-curr_int, curr_int))


class SpriteSheetLoader:
    def __init__(self, path, is_traditional=False):
        self.sprites = {}
        if os.path.exists(path):
            sheet = pygame.image.load(path).convert_alpha()
            if is_traditional:
                # Traditional sheet: rows = H, S, D, C; cols 2-14 = A-K; cols 0-1 = joker/back
                sheet_w, sheet_h = sheet.get_size()
                card_w = sheet_w // 15   # 15 columns total
                card_h = sheet_h // 6    # 6 rows (4 standard + 2 extra color variants)
                suit_map = {0: 'H', 1: 'S', 2: 'D', 3: 'C'}
                for row, suit in suit_map.items():
                    for col_idx in range(13):  # A through K
                        key = f"{suit}{col_idx+1}"
                        src_col = col_idx + 2  # Skip joker + back columns
                        rect = pygame.Rect(src_col * card_w, row * card_h, card_w, card_h)
                        surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                        surf.blit(sheet, (0, 0), rect)
                        self.sprites[key] = pygame.transform.smoothscale(surf, (75, 108))
                # Card back is at column 1, row 1 (Spades row)
                back_rect = pygame.Rect(1 * card_w, 1 * card_h, card_w, card_h)
                back_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                back_surf.blit(sheet, (0, 0), back_rect)
                self.card_back = pygame.transform.smoothscale(back_surf, (75, 108))
            else:
                # CuteCards format: rows = C, D, S, H; cols 0-12 = A-K
                suit_map = {0: 'C', 1: 'D', 2: 'S', 3: 'H'}
                card_w, card_h = 100, 144
                for row, suit in suit_map.items():
                    for col in range(13):
                        key = f"{suit}{col+1}"
                        rect = pygame.Rect(col * card_w, row * card_h, card_w, card_h)
                        surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                        surf.blit(sheet, (0, 0), rect)
                        self.sprites[key] = pygame.transform.smoothscale(surf, (75, 108))
                back_rect = pygame.Rect(14 * card_w, 2 * card_h, card_w, card_h)
                back_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                back_surf.blit(sheet, (0, 0), back_rect)
                self.card_back = pygame.transform.smoothscale(back_surf, (75, 108))
        else:
            self.card_back = pygame.Surface((75, 108))
            self.card_back.fill((45, 75, 160))

    def get_card(self, code: str):
        return self.sprites.get(code, self.card_back)


def draw_button(surface, text, rect, base_color, mouse_pos):
    hover = rect.collidepoint(mouse_pos)
    color = (min(base_color[0]+25, 255), min(base_color[1]+25, 255), min(base_color[2]+25, 255)) if hover else base_color
    # Soft shadow
    shadow_rect = rect.move(2, 2)
    pygame.draw.rect(surface, (0, 0, 0, 60) if hover else (0, 40, 0), shadow_rect, border_radius=10)
    pygame.draw.rect(surface, color, rect, border_radius=10)
    pygame.draw.rect(surface, GOLD, rect, 2, border_radius=10)
    txt_surf = FONT_MED.render(text, True, WHITE)
    surface.blit(txt_surf, (rect.centerx - txt_surf.get_width() // 2, rect.centery - txt_surf.get_height() // 2))


def show_rules_dialog():
    """Displays rules dialog box."""
    dialog_open = True
    while dialog_open:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                dialog_open = False

        draw_felt_background(SCREEN, WIDTH, HEIGHT)
        overlay = pygame.Rect(100, 80, WIDTH - 200, HEIGHT - 160)
        pygame.draw.rect(SCREEN, SLOT_BG, overlay, border_radius=16)
        pygame.draw.rect(SCREEN, GOLD, overlay, 3, border_radius=16)

        rules = [
            "GIN RUMMY RULES & SCORING",
            "1. Objective: Form Melds (Sets of 3/4 same rank or Runs of 3+ same suit).",
            "2. Draw a card each turn from Deck or Discard, then Discard one card.",
            "3. Knocking: Knock when Deadwood points are 10 or lower.",
            "4. Scoring System:",
            "   • Normal Knock: Difference in Deadwood (Knocker wins).",
            "   • Gin (0 Deadwood): Opponent Deadwood + 25 Bonus points.",
            "   • Undercut: Opponent has <= Deadwood. Opponent gains difference + 25 Bonus.",
            "5. First to 100 points wins the match!",
            "",
            "[ Click or Press Any Key to Return to Menu ]"
        ]

        for i, line in enumerate(rules):
            color = GOLD if i in [0, 4] else WHITE
            font = FONT_LARGE if i == 0 else FONT_MED
            t_surf = font.render(line, True, color)
            SCREEN.blit(t_surf, (140, 110 + i * 40))

        pygame.display.flip()


def show_game_over_dialog(user_score: int, ai_score: int):
    """Displays the game over screen with match results."""
    dialog_open = True
    user_won = user_score >= 100
    
    while dialog_open:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dialog_open = False

        draw_felt_background(SCREEN, WIDTH, HEIGHT)
        overlay = pygame.Rect(180, 120, WIDTH - 360, HEIGHT - 240)
        pygame.draw.rect(SCREEN, SLOT_BG, overlay, border_radius=16)
        
        border_color = GOLD if user_won else BTN_RED
        pygame.draw.rect(SCREEN, border_color, overlay, 3, border_radius=16)

        header_txt = "VICTORY!" if user_won else "DEFEAT!"
        header_color = GOLD if user_won else BTN_RED
        
        t_header = FONT_TITLE.render(header_txt, True, header_color)
        t_sub = FONT_LARGE.render("FINAL MATCH RESULTS", True, WHITE)
        t_user = FONT_LARGE.render(f"YOUR SCORE: {user_score}", True, GOLD)
        t_ai = FONT_LARGE.render(f"AI SCORE: {ai_score}", True, WHITE)
        t_prompt = FONT_MED.render("[ Click anywhere to return to Main Menu ]", True, GOLD)

        SCREEN.blit(t_header, (WIDTH // 2 - t_header.get_width() // 2, 160))
        SCREEN.blit(t_sub, (WIDTH // 2 - t_sub.get_width() // 2, 230))
        SCREEN.blit(t_user, (WIDTH // 2 - t_user.get_width() // 2, 310))
        SCREEN.blit(t_ai, (WIDTH // 2 - t_ai.get_width() // 2, 360))
        SCREEN.blit(t_prompt, (WIDTH // 2 - t_prompt.get_width() // 2, 460))

        pygame.display.flip()


class GinRummyGUI:
    def __init__(self):
        self.sounds = SoundManager()
        self.shake = ScreenShake()
        self.theme_idx = 0
        self.loaders = [SpriteSheetLoader(path, is_traditional=(i == 3)) for i, (_, path) in enumerate(CARD_THEMES)]
        self.game = Play()
        self.active_animations = []
        self.gin_celebration_ticks = 0

    @property
    def current_loader(self):
        return self.loaders[self.theme_idx]

    @property
    def current_theme_name(self):
        return CARD_THEMES[self.theme_idx][0]

    def cycle_theme(self):
        self.theme_idx = (self.theme_idx + 1) % len(CARD_THEMES)
        self.sounds.play("click")

    def main_menu(self):
        """Main menu screen."""
        clock = pygame.time.Clock()
        in_menu = True

        while in_menu:
            m_pos = pygame.mouse.get_pos()
            draw_felt_background(SCREEN, WIDTH, HEIGHT)
            # Central lighter felt circle
            circle_surf = pygame.Surface((760, 760), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (20, 130, 20, 40), (380, 380), 380)
            SCREEN.blit(circle_surf, (WIDTH // 2 - 380, HEIGHT // 2 - 380))

            title = FONT_TITLE.render("GIN RUMMY AI", True, GOLD)
            subtitle = FONT_MED.render("Programming for AI Project", True, WHITE)
            theme_hint = FONT_SMALL.render(f"Card Theme: {self.current_theme_name}", True, GOLD)
            SCREEN.blit(title, (WIDTH // 2 - title.get_width() // 2, 120))
            SCREEN.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 185))
            SCREEN.blit(theme_hint, (WIDTH // 2 - theme_hint.get_width() // 2, 220))

            btn_start = pygame.Rect(WIDTH // 2 - 130, 290, 260, 55)
            btn_theme = pygame.Rect(WIDTH // 2 - 130, 365, 260, 55)
            btn_rules = pygame.Rect(WIDTH // 2 - 130, 440, 260, 55)
            music_label = "MUSIC: ON" if self.sounds.bgm_playing else "MUSIC: OFF"
            btn_music = pygame.Rect(WIDTH // 2 - 130, 515, 260, 45)

            draw_button(SCREEN, "START GAME", btn_start, BTN_GREEN, m_pos)
            draw_button(SCREEN, f"THEME: {self.current_theme_name}", btn_theme, BTN_DARK, m_pos)
            draw_button(SCREEN, "RULES / HOW TO PLAY", btn_rules, BTN_DARK, m_pos)
            draw_button(SCREEN, music_label, btn_music, (60, 60, 70), m_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_start.collidepoint(m_pos):
                        self.sounds.play("click")
                        self.game = Play()
                        in_menu = False
                    elif btn_theme.collidepoint(m_pos):
                        self.cycle_theme()
                    elif btn_rules.collidepoint(m_pos):
                        self.sounds.play("click")
                        show_rules_dialog()
                    elif btn_music.collidepoint(m_pos):
                        self.sounds.play("click")
                        self.sounds.toggle_music()

            pygame.display.flip()
            clock.tick(30)

    def run_game_loop(self):
        """Active gameplay session UI loop with drag-and-drop, animations, sound effects, and screen shake."""
        clock = pygame.time.Clock()
        running = True
        game_surf = pygame.Surface((WIDTH, HEIGHT))

        # Drag state
        dragging = False
        drag_card_code = None
        drag_source = None    # "hand", "deck", "discard"
        drag_origin = (0, 0)  # Original position for snap-back
        drag_offset = (0, 0)  # Mouse offset from card top-left
        drag_is_face_up = True

        # Hand area zone for drop detection (where player's cards are)
        HAND_ZONE_Y = HEIGHT - 220

        while running:
            m_pos = pygame.mouse.get_pos()
            draw_felt_background(game_surf, WIDTH, HEIGHT)

            # Deadwood & Knock Calculation
            cards = self.game.user_hand.sort_cards_by_mode(self.game.user_sort_mode)
            if len(cards) == 11 and self.game.current_phase == "DISCARD":
                best_disc, user_dw = self.game.user_hand.get_best_knock_discard()
            else:
                user_dw = self.game.user_hand.calculate_deadwood()
                best_disc = None

            # Buttons
            btn_surrender = pygame.Rect(30, HEIGHT - 70, 140, 45)
            btn_theme = pygame.Rect(WIDTH - 365, 15, 175, 35)
            btn_sort = pygame.Rect(WIDTH - 180, 15, 150, 35)
            music_icon = "MUSIC: ON" if self.sounds.bgm_playing else "MUSIC: OFF"
            btn_music_game = pygame.Rect(30, 60, 120, 30)
            btn_knock = pygame.Rect(WIDTH - 170, HEIGHT - 70, 140, 45)
            btn_next = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 70, 200, 45)

            draw_button(game_surf, "SURRENDER", btn_surrender, (160, 40, 30), m_pos)
            draw_button(game_surf, f"THEME: {self.current_theme_name}", btn_theme, BTN_DARK, m_pos)
            draw_button(game_surf, f"SORT: {self.game.user_sort_mode}", btn_sort, BTN_DARK, m_pos)
            draw_button(game_surf, music_icon, btn_music_game, (60, 60, 70), m_pos)

            # Show dynamic GIN! or KNOCK Button
            can_knock = (self.game.current_turn == "USER" and 
                         self.game.current_phase == "DISCARD" and 
                         user_dw <= 10 and 
                         not self.game.round_over)
            if can_knock:
                btn_label = "GIN!" if user_dw == 0 else "KNOCK"
                btn_color = (200, 160, 20) if user_dw == 0 else BTN_GREEN
                draw_button(game_surf, btn_label, btn_knock, btn_color, m_pos)

            # Show NEXT ROUND button when round finishes
            if self.game.round_over:
                draw_button(game_surf, "NEXT ROUND", btn_next, BTN_GREEN, m_pos)

            # Table Slots
            deck_rect = pygame.Rect(WIDTH // 2 - 90, HEIGHT // 2 - 54, 75, 108)
            disc_rect = pygame.Rect(WIDTH // 2 + 15, HEIGHT // 2 - 54, 75, 108)
            pygame.draw.rect(game_surf, SLOT_BG, deck_rect, border_radius=8)
            pygame.draw.rect(game_surf, SLOT_BG, disc_rect, border_radius=8)
            # Slot labels
            deck_label = FONT_SMALL.render("DECK", True, (100, 180, 100))
            disc_label = FONT_SMALL.render("DISCARD", True, (100, 180, 100))
            game_surf.blit(deck_label, (deck_rect.centerx - deck_label.get_width() // 2, deck_rect.bottom + 4))
            game_surf.blit(disc_label, (disc_rect.centerx - disc_label.get_width() // 2, disc_rect.bottom + 4))

            if self.game.deck_remaining > 0:
                game_surf.blit(self.current_loader.card_back, deck_rect.topleft)
            top_disc = self.game.discard_pile.top()
            if top_disc and not (dragging and drag_source == "discard"):
                game_surf.blit(self.current_loader.get_card(top_disc), disc_rect.topleft)

            # AI Hand (Exposed face-up when round is over)
            ai_cards = self.game.ai_hand.get_cards()
            start_x_ai = WIDTH // 2 - (len(ai_cards) * 22)
            for i, code in enumerate(ai_cards):
                card_img = self.current_loader.get_card(code) if self.game.round_over else self.current_loader.card_back
                game_surf.blit(card_img, (start_x_ai + i * 44, 35))

            # Player Hand (skip the card being dragged)
            card_rects = []
            start_x = WIDTH // 2 - (len(cards) * 26)
            for i, code in enumerate(cards):
                rect = pygame.Rect(start_x + i * 52, HEIGHT - 180, 75, 108)
                if not (dragging and drag_source == "hand" and drag_card_code == code):
                    game_surf.blit(self.current_loader.get_card(code), rect.topleft)
                card_rects.append((rect, code))

            # Drop zone highlight when dragging
            if dragging:
                if drag_source == "hand" and self.game.current_phase == "DISCARD":
                    # Highlight discard pile as drop target
                    highlight = pygame.Surface((75 + 10, 108 + 10), pygame.SRCALPHA)
                    glow_color = (255, 215, 0, 60) if disc_rect.inflate(30, 30).collidepoint(m_pos) else (255, 255, 255, 30)
                    highlight.fill(glow_color)
                    game_surf.blit(highlight, (disc_rect.x - 5, disc_rect.y - 5))
                elif drag_source in ("deck", "discard") and self.game.current_phase == "DRAW":
                    # Highlight hand zone as drop target
                    hand_highlight = pygame.Surface((WIDTH, 200), pygame.SRCALPHA)
                    glow_color = (255, 215, 0, 30) if m_pos[1] > HAND_ZONE_Y else (255, 255, 255, 15)
                    hand_highlight.fill(glow_color)
                    game_surf.blit(hand_highlight, (0, HEIGHT - 200))

            # Render Active Flying Animations
            for anim in self.active_animations[:]:
                anim_pos = anim.get_pos()
                card_img = self.current_loader.get_card(anim.card_code) if anim.is_face_up else self.current_loader.card_back
                # Subtle drop shadow for flying card
                shadow = pygame.Surface((75, 108), pygame.SRCALPHA)
                shadow.fill((0, 0, 0, 80))
                game_surf.blit(shadow, (anim_pos[0] + 5, anim_pos[1] + 5))
                game_surf.blit(card_img, anim_pos)
                if anim.finished:
                    self.active_animations.remove(anim)

            # Render Dragged Card (on top of everything)
            if dragging and drag_card_code:
                drag_x = m_pos[0] - drag_offset[0]
                drag_y = m_pos[1] - drag_offset[1]
                card_img = self.current_loader.get_card(drag_card_code) if drag_is_face_up else self.current_loader.card_back
                # Drop shadow under dragged card
                shadow = pygame.Surface((75, 108), pygame.SRCALPHA)
                shadow.fill((0, 0, 0, 90))
                game_surf.blit(shadow, (drag_x + 6, drag_y + 6))
                game_surf.blit(card_img, (drag_x, drag_y))

            # HUD Display
            hud = FONT_MED.render(
                f"YOU: {self.game.user_score} PTS | AI: {self.game.ai_score} PTS | Deck: {self.game.deck_remaining} | AI Acc: {self.game.ai_engine.accuracy:.1f}%", 
                True, GOLD
            )
            dw_msg = f"Your Deadwood: {user_dw}" + (f" (discarding {best_disc})" if best_disc and can_knock else "")
            dw_txt = FONT_SMALL.render(dw_msg, True, WHITE)

            # Drag hint message
            if self.game.current_turn == "USER" and not self.game.round_over and not dragging:
                if self.game.current_phase == "DRAW":
                    hint = "Drag a card from Deck or Discard pile to your hand"
                elif self.game.current_phase == "DISCARD":
                    hint = "Drag a card from your hand to the Discard pile"
                else:
                    hint = self.game.status_msg
                msg = FONT_MED.render(hint, True, WHITE)
            else:
                msg = FONT_MED.render(self.game.status_msg, True, WHITE)

            game_surf.blit(hud, (30, 15))
            game_surf.blit(dw_txt, (30, 42))
            game_surf.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT - 220))

            # Gin Celebration Overlay Banner
            if pygame.time.get_ticks() < self.gin_celebration_ticks:
                gin_banner = pygame.Rect(WIDTH // 2 - 250, HEIGHT // 2 - 50, 500, 100)
                pygame.draw.rect(game_surf, (20, 20, 25, 230), gin_banner, border_radius=16)
                pygame.draw.rect(game_surf, GOLD, gin_banner, 4, border_radius=16)
                t_gin = FONT_TITLE.render("★ GIN! +25 BONUS ★", True, GOLD)
                game_surf.blit(t_gin, (WIDTH // 2 - t_gin.get_width() // 2, HEIGHT // 2 - 25))

            # Screen Shake Blit
            offset_x, offset_y = self.shake.get_offset()
            SCREEN.fill((0, 0, 0))
            SCREEN.blit(game_surf, (offset_x, offset_y))
            pygame.display.flip()

            # Check 100+ points match end trigger
            if self.game.round_over and (self.game.user_score >= 100 or self.game.ai_score >= 100):
                pygame.time.delay(1000)
                show_game_over_dialog(self.game.user_score, self.game.ai_score)
                return

            # Execute AI Turn via play.py
            if self.game.current_turn == "AI" and not self.game.round_over:
                pygame.time.delay(500)
                # Capture state before turn
                ai_cards = self.game.ai_hand.get_cards()
                top_disc_before = self.game.discard_pile.top()
                take_disc = self.game.ai_engine.predict_draw(self.game.turn_number, self.game.deck_remaining, ai_cards, top_disc_before)

                start_pt = disc_rect.topleft if (take_disc and top_disc_before) else deck_rect.topleft
                self.sounds.play("draw")
                self.active_animations.append(CardAnimation(None, start_pt, (WIDTH // 2, 35), 200, is_face_up=False))

                # Execute AI
                self.game.execute_ai_turn()

                # Discard animation for AI
                new_top_disc = self.game.discard_pile.top()
                if new_top_disc:
                    self.sounds.play("discard")
                    self.active_animations.append(CardAnimation(new_top_disc, (WIDTH // 2, 35), disc_rect.topleft, 200, is_face_up=True))

                # Check if AI Knocked or Ginned
                if self.game.round_over:
                    ai_dw = self.game.ai_hand.calculate_deadwood()
                    if ai_dw == 0:
                        self.sounds.play("gin")
                        self.shake.start(duration_ms=650, intensity=18)
                        self.gin_celebration_ticks = pygame.time.get_ticks() + 1500
                    else:
                        self.sounds.play("knock")
                        self.shake.start(duration_ms=450, intensity=12)

            # Input Events (Drag-and-Drop)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Button clicks (non-drag)
                    if btn_sort.collidepoint(m_pos):
                        self.sounds.play("click")
                        self.game.cycle_sort_mode()
                    elif btn_theme.collidepoint(m_pos):
                        self.cycle_theme()
                    elif btn_music_game.collidepoint(m_pos):
                        self.sounds.play("click")
                        self.sounds.toggle_music()
                    elif btn_surrender.collidepoint(m_pos):
                        self.sounds.play("click")
                        self.game.handle_surrender()
                    elif self.game.round_over and btn_next.collidepoint(m_pos):
                        self.sounds.play("click")
                        self.game.start_new_round()
                    elif can_knock and btn_knock.collidepoint(m_pos):
                        if user_dw == 0:
                            self.sounds.play("gin")
                            self.shake.start(duration_ms=650, intensity=18)
                            self.gin_celebration_ticks = pygame.time.get_ticks() + 1500
                        else:
                            self.sounds.play("knock")
                            self.shake.start(duration_ms=450, intensity=12)
                        self.game.handle_user_knock(best_disc)

                    # Start drag from player hand (DISCARD phase)
                    elif (self.game.current_turn == "USER" and not self.game.round_over 
                          and self.game.current_phase == "DISCARD"):
                        for rect, code in reversed(card_rects):
                            if rect.collidepoint(m_pos):
                                dragging = True
                                drag_card_code = code
                                drag_source = "hand"
                                drag_origin = rect.topleft
                                drag_offset = (m_pos[0] - rect.x, m_pos[1] - rect.y)
                                drag_is_face_up = True
                                break

                    # Start drag from deck or discard pile (DRAW phase)
                    elif (self.game.current_turn == "USER" and not self.game.round_over 
                          and self.game.current_phase == "DRAW"):
                        if deck_rect.collidepoint(m_pos) and self.game.deck_remaining > 0:
                            dragging = True
                            drag_card_code = None  # Face-down, unknown
                            drag_source = "deck"
                            drag_origin = deck_rect.topleft
                            drag_offset = (m_pos[0] - deck_rect.x, m_pos[1] - deck_rect.y)
                            drag_is_face_up = False
                        elif disc_rect.collidepoint(m_pos) and self.game.discard_pile.top():
                            dragging = True
                            drag_card_code = self.game.discard_pile.top()
                            drag_source = "discard"
                            drag_origin = disc_rect.topleft
                            drag_offset = (m_pos[0] - disc_rect.x, m_pos[1] - disc_rect.y)
                            drag_is_face_up = True

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if dragging:
                        # Check if dropped in valid zone
                        if drag_source == "hand" and self.game.current_phase == "DISCARD":
                            # Drop on discard pile?
                            drop_zone = disc_rect.inflate(40, 40)
                            if drop_zone.collidepoint(m_pos):
                                self.sounds.play("discard")
                                self.active_animations.append(
                                    CardAnimation(drag_card_code, (m_pos[0] - drag_offset[0], m_pos[1] - drag_offset[1]), 
                                                  disc_rect.topleft, 120, is_face_up=True))
                                self.game.handle_user_discard(drag_card_code)
                            # else: card snaps back (just stop dragging)

                        elif drag_source == "deck" and self.game.current_phase == "DRAW":
                            # Drop in hand zone?
                            if m_pos[1] > HAND_ZONE_Y:
                                self.sounds.play("draw")
                                dest_x = start_x + len(cards) * 52
                                self.active_animations.append(
                                    CardAnimation(None, (m_pos[0] - drag_offset[0], m_pos[1] - drag_offset[1]),
                                                  (dest_x, HEIGHT - 180), 150, is_face_up=False))
                                self.game.handle_user_draw_deck()

                        elif drag_source == "discard" and self.game.current_phase == "DRAW":
                            # Drop in hand zone?
                            if m_pos[1] > HAND_ZONE_Y:
                                self.sounds.play("draw")
                                dest_x = start_x + len(cards) * 52
                                self.active_animations.append(
                                    CardAnimation(drag_card_code, (m_pos[0] - drag_offset[0], m_pos[1] - drag_offset[1]),
                                                  (dest_x, HEIGHT - 180), 150, is_face_up=True))
                                self.game.handle_user_draw_discard()

                        # Reset drag state
                        dragging = False
                        drag_card_code = None
                        drag_source = None

            clock.tick(30)
        pygame.quit()

    def run(self):
        while True:
            self.main_menu()
            self.run_game_loop()

