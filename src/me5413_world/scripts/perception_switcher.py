#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perception_switcher.py

Listens to the cone trigger topic (default: /cmd_unblock):
  - Only accepts the first valid True signal; all subsequent signals are ignored.
  - On trigger:
      1. Snapshots and locks the current counts from /percep/numbers
      2. Shuts down the box_counter_perception node
      3. This node takes over the enter_room logic (uses locked counts to determine the minimum value)
"""

import json
import threading
import subprocess
import rospy
from std_msgs.msg import Bool, String, Int32


class PerceptionSwitcher:
    def __init__(self):
        # ===== Parameters =====
        self.cone_trigger_topic      = rospy.get_param("~cone_trigger_topic",      "/cmd_unblock")
        self.box_counter_node_name   = rospy.get_param("~box_counter_node_name",   "/box_counter_perception")
        self.records_topic           = rospy.get_param("~records_topic",           "/percep/numbers")
        self.current_seen_digit_topic= rospy.get_param("~current_seen_digit_topic","/percep/current_seen_digit")
        self.enter_room_topic        = rospy.get_param("~enter_room_topic",        "/second_floor/enter_room")
        self.enter_trigger_topic     = rospy.get_param("~enter_trigger_topic",     "/second_floor/enter_room_trigger")

        # When ties are not allowed, only the smallest digit in the minimum-count set is used
        self.allow_tie               = rospy.get_param("~allow_tie",               True)
        # Cooldown (seconds) to suppress repeated triggers for the same digit
        self.same_digit_cooldown     = rospy.get_param("~same_digit_cooldown",     2.0)
        # Whether to exclude digits with count=0 from the minimum-value competition
        self.ignore_zero_count       = rospy.get_param("~ignore_zero_count",       True)
        # Digit → room ID mapping
        self.room_map                = rospy.get_param("~room_map", {
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
            "5": 5, "6": 6, "7": 7, "8": 8, "9": 9
        })

        # ===== State =====
        self._lock          = threading.Lock()
        self._switched      = False          # Whether the switch has been completed
        self._live_counts   = {i: 0 for i in range(10)}   # Live counts from box_counter
        self._locked_counts = None           # Snapshot taken at switch time; used exclusively afterwards
        self._last_trigger_time = {i: -1e9 for i in range(10)}

        # ===== Publishers =====
        self._enter_room_pub    = rospy.Publisher(self.enter_room_topic,    Int32, queue_size=1)
        self._enter_trigger_pub = rospy.Publisher(self.enter_trigger_topic, Bool,  queue_size=1)
        self.freeze_counts_topic = rospy.get_param("~freeze_counts_topic", "/percep/freeze_counts")
        self._freeze_pub = rospy.Publisher(self.freeze_counts_topic, Bool, queue_size=1, latch=True)

        # ===== Subscribers =====
        # Always subscribe to counts so the latest value is available at switch time
        self._records_sub = rospy.Subscriber(
            self.records_topic, String, self._records_cb, queue_size=1
        )
        # Unblock trigger (only the first True is acted upon)
        self._trigger_sub = rospy.Subscriber(
            self.cone_trigger_topic, Bool, self._trigger_cb, queue_size=1
        )
        # Currently seen digit (ignored until after the switch)
        self._digit_sub = rospy.Subscriber(
            self.current_seen_digit_topic, Int32, self._digit_cb, queue_size=1
        )

        rospy.loginfo(
            "[PerceptionSwitcher] Ready. Listening on '%s'.",
            self.cone_trigger_topic
        )

    # ------------------------------------------------------------------
    # Live count updates (before the switch)
    # ------------------------------------------------------------------
    def _records_cb(self, msg):
        if self._switched:
            return  # After the switch, live counts are no longer updated; the locked snapshot is used instead
        try:
            data = json.loads(msg.data)
            counts_raw = data.get("counts", {})
            new_counts = {i: 0 for i in range(10)}
            for k, v in counts_raw.items():
                try:
                    d = int(k)
                    if 0 <= d <= 9:
                        new_counts[d] = int(v)
                except Exception:
                    pass
            with self._lock:
                self._live_counts = new_counts
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[PerceptionSwitcher] Failed to parse counts: %s", str(e))

    # ------------------------------------------------------------------
    # Unblock trigger callback: only the first signal is processed
    # ------------------------------------------------------------------
    def _trigger_cb(self, msg):
        if hasattr(msg, 'data') and not msg.data:
            return

        with self._lock:
            if self._switched:
                rospy.logwarn_throttle(
                    5.0,
                    "[PerceptionSwitcher] Ignoring subsequent unblock signal (already switched)."
                )
                return
            self._switched = True
            # Lock the current counts snapshot
            self._locked_counts = dict(self._live_counts)

        rospy.logwarn(
            "[PerceptionSwitcher] Unblock received! Locked counts: %s",
            str(self._locked_counts)
        )

        rospy.sleep(0.2)
        self._freeze_pub.publish(Bool(data=True))
        rospy.logwarn("[PerceptionSwitcher] Sent freeze signal on %s.", self.freeze_counts_topic)

    # def _freeze_box_counter(self):
    #     freeze_topic = rospy.get_param("~freeze_counts_topic", "/percep/freeze_counts")
    #     pub = rospy.Publisher(freeze_topic, Bool, queue_size=1, latch=True)
    #     rospy.sleep(0.3)  # Wait for the publisher connection to be established
    #     pub.publish(Bool(data=True))
    #     rospy.logwarn("[PerceptionSwitcher] Sent freeze signal to box_counter.")

    # ------------------------------------------------------------------
    # Currently seen digit → determine whether it matches the minimum
    #   count using the locked snapshot
    # ------------------------------------------------------------------
    def _digit_cb(self, msg):
        # Do not process until the switch is complete
        if not self._switched or self._locked_counts is None:
            return

        digit = int(msg.data)
        if digit < 0 or digit > 9:
            return

        min_digits = self._get_min_digits()
        if not min_digits:
            return

        if not self.allow_tie:
            min_digits = [min(min_digits)]

        if digit not in min_digits:
            rospy.loginfo_throttle(
                1.0,
                "[PerceptionSwitcher] Seen digit=%d not in min_digits=%s (locked counts=%s)",
                digit, str(min_digits), str(self._locked_counts)
            )
            return

        now = rospy.Time.now().to_sec()
        if now - self._last_trigger_time[digit] < self.same_digit_cooldown:
            return

        self._last_trigger_time[digit] = now
        room_id = int(self.room_map.get(str(digit), digit))

        self._enter_room_pub.publish(Int32(data=room_id))
        self._enter_trigger_pub.publish(Bool(data=True))

        rospy.logwarn(
            "[PerceptionSwitcher] Enter room: digit=%d → room_id=%d | locked counts=%s | min_digits=%s",
            digit, room_id, str(self._locked_counts), str(min_digits)
        )

    def _get_min_digits(self):
        """Returns the list of digits with the lowest count from the locked snapshot."""
        items = []
        for d in range(10):
            c = self._locked_counts.get(d, 0)
            if self.ignore_zero_count and c == 0:
                continue
            items.append((d, c))
        if not items:
            return []
        min_c = min(c for _, c in items)
        return [d for d, c in items if c == min_c]


if __name__ == "__main__":
    rospy.init_node("perception_switcher")
    node = PerceptionSwitcher()
    rospy.spin()