from typing import Optional
import pygame

from pong.agent import UserAgent

from .environment import Action, Ball, Environment, Field, Player

white = (255, 255, 255)
gray = (128, 128, 128)
black = (0, 0, 0)
red = (255, 0, 0)


class Renderer(object):
    def __init__(self, screen_width: int, screen_height: int, env: Environment) -> None:
        """PyGame renderer for the pong environment.

        Args:
            screen_width (int): screen width
            screen_height (int): screen height
            env (Environment): pong environment
        """
        pygame.init()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.user_agent = UserAgent()

        self.font = pygame.font.SysFont(None, 72)

        pygame.display.set_caption("Pong by Kinguuu")

        self.game_over = False
        self.env = env

        self.m2p = self.screen_width / (env.field.width - env.field.origin[0])

        self.clock = pygame.time.Clock()

    def draw_player(self, p: Player, upscale: float = 1.0) -> None:
        """Draw a player to the screen.

        Args:
            p (Player): player to draw
            upscale (float, optional): scale to screen res. Defaults to 1.0.
        """
        x = upscale * p.pos_x
        y = upscale * p.pos_y
        w = upscale * p.width
        h = upscale * p.height
        pygame.draw.rect(self.screen, white, [x, y, w, h])

    def draw_ball(self, ball: Ball, upscale: float = 1.0) -> None:
        """Draw ball to the screen.

        Args:
            ball (Ball): ball to draw
            upscale (float, optional): scale to screen res. Defaults to 1.0.
        """
        pos = upscale * ball.pos
        rad = upscale * ball.radius
        pygame.draw.circle(self.screen, white, pos, rad)

    def draw_field(self, field: Field, upscale: float = 1.0) -> None:
        """Draw field to screen.

        Args:
            field (Field): the field to draw
            upscale (float, optional): scale to screen res. Defaults to 1.0.
        """
        x_center = upscale * (field.origin[0] + 0.5 * field.width)
        self.screen.fill(black)
        pygame.draw.line(
            self.screen,
            gray,
            (x_center, upscale * field.origin[1]),
            (x_center, upscale * (field.origin[1] + field.height)),
        )

    def draw_score(self, upscale: float = 1) -> None:
        """Draw score to screen.

        Args:
            upscale (float, optional): scale to screen res. Defaults to 1.
        """
        score = self.env.states[-1]["score"].astype(int)
        img = self.font.render(f"{score[0]} : {score[1]}", True, gray, black)
        x_pos = (
            self.env.field.origin[0]
            + (self.env.field.width - 0.25 * img.get_bounding_rect().width)
            * 0.5
            * upscale
        )
        y_pos = self.env.field.height * upscale * 0.05
        self.screen.blit(img, (x_pos, y_pos))

    def render(self, framerate: int = 0) -> None:
        """Render the screen with passed framerate.

        Args:
            framerate (int, optional): Defaults to 0.
        """
        self.draw_field(self.env.field, upscale=self.m2p)
        self.draw_player(self.env.p1, upscale=self.m2p)
        self.draw_player(self.env.p2, upscale=self.m2p)
        self.draw_score(self.m2p)
        self.draw_ball(self.env.ball, upscale=self.m2p)

        pygame.display.update()
        self.clock.tick(framerate)

    def events(self, user_control: bool = False) -> Optional[Action]:
        """Event handling for user interactions.
        
        If user_control is set to True, user inputs get returned as actions.

        Args:
            user_control (bool, optional): Defaults to False.

        Returns:
            Optional[Action]: user action or None.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if user_control:
                return self.user_agent.select_action(state=None, event=event)

        return self.user_agent.action

    def quit(self) -> None:
        pygame.quit()
