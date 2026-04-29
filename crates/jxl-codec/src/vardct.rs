use crate::decode::ImageRegion;
use crate::frame::{FrameEncoding, FrameHeader};
use crate::frame_data::{FrameData, FrameSectionKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctFrameMetadata {
    pub width: u32,
    pub height: u32,
    pub group_dim: u32,
    pub groups_x: u32,
    pub groups_y: u32,
    pub dc_groups_x: u32,
    pub dc_groups_y: u32,
    pub is_combined: bool,
    pub global_section: Option<VarDctSectionMetadata>,
    pub ac_global_section: Option<VarDctSectionMetadata>,
    pub sections: Vec<VarDctSectionMetadata>,
    pub ac_groups: Vec<VarDctGroupMetadata>,
    pub dc_groups: Vec<VarDctGroupMetadata>,
    pub ac_group_sections: Vec<VarDctPassGroupSectionMetadata>,
    pub dc_group_sections: Vec<VarDctGroupSectionMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctSectionMetadata {
    pub section_logical_id: usize,
    pub section_physical_index: usize,
    pub section_kind: FrameSectionKind,
    pub codestream_offset: usize,
    pub payload_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarDctGroupMetadata {
    pub group: usize,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctGroupSectionMetadata {
    pub section: VarDctSectionMetadata,
    pub group: VarDctGroupMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctPassGroupSectionMetadata {
    pub section: VarDctSectionMetadata,
    pub pass: usize,
    pub group: VarDctGroupMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VarDctSectionBuckets {
    is_combined: bool,
    global_section: Option<VarDctSectionMetadata>,
    ac_global_section: Option<VarDctSectionMetadata>,
    ac_group_sections: Vec<VarDctPassGroupSectionMetadata>,
    dc_group_sections: Vec<VarDctGroupSectionMetadata>,
}

impl VarDctFrameMetadata {
    pub fn ac_groups_intersecting_region(&self, region: ImageRegion) -> Vec<usize> {
        self.ac_groups
            .iter()
            .filter(|group| group_intersects_region(group, region))
            .map(|group| group.group)
            .collect()
    }

    pub fn ac_sections_for_region(
        &self,
        region: ImageRegion,
    ) -> Vec<&VarDctPassGroupSectionMetadata> {
        if self.is_combined {
            return Vec::new();
        }
        self.ac_group_sections
            .iter()
            .filter(|section| group_intersects_region(&section.group, region))
            .collect()
    }

    pub fn dc_sections_for_region(&self, region: ImageRegion) -> Vec<&VarDctGroupSectionMetadata> {
        if self.is_combined {
            return Vec::new();
        }
        self.dc_group_sections
            .iter()
            .filter(|section| group_intersects_region(&section.group, region))
            .collect()
    }
}

pub fn read_vardct_frame_metadata(
    frame_header: &FrameHeader,
    frame_data: &FrameData,
) -> Option<VarDctFrameMetadata> {
    if frame_header.encoding != FrameEncoding::VarDct {
        return None;
    }

    let sections = frame_data
        .sections
        .iter()
        .map(|section| VarDctSectionMetadata {
            section_logical_id: section.logical_id,
            section_physical_index: section.physical_index,
            section_kind: section.kind,
            codestream_offset: section.codestream_offset,
            payload_size: section.size,
        })
        .collect::<Vec<_>>();
    let ac_groups = group_metadata(
        frame_header.group_layout.groups_x,
        frame_header.group_layout.groups_y,
        frame_header.group_layout.group_dim,
        frame_header.frame_size.width,
        frame_header.frame_size.height,
    );
    let dc_groups = group_metadata(
        frame_header.group_layout.dc_groups_x,
        frame_header.group_layout.dc_groups_y,
        frame_header.group_layout.dc_group_dim,
        frame_header.frame_size.width,
        frame_header.frame_size.height,
    );
    let buckets = classify_vardct_sections(&sections, &ac_groups, &dc_groups);

    Some(VarDctFrameMetadata {
        width: frame_header.frame_size.width,
        height: frame_header.frame_size.height,
        group_dim: frame_header.group_layout.group_dim,
        groups_x: frame_header.group_layout.groups_x,
        groups_y: frame_header.group_layout.groups_y,
        dc_groups_x: frame_header.group_layout.dc_groups_x,
        dc_groups_y: frame_header.group_layout.dc_groups_y,
        is_combined: buckets.is_combined,
        global_section: buckets.global_section,
        ac_global_section: buckets.ac_global_section,
        sections,
        ac_groups,
        dc_groups,
        ac_group_sections: buckets.ac_group_sections,
        dc_group_sections: buckets.dc_group_sections,
    })
}

fn classify_vardct_sections(
    sections: &[VarDctSectionMetadata],
    ac_groups: &[VarDctGroupMetadata],
    dc_groups: &[VarDctGroupMetadata],
) -> VarDctSectionBuckets {
    let global_section = sections
        .iter()
        .find(|section| {
            matches!(
                section.section_kind,
                FrameSectionKind::Combined | FrameSectionKind::DcGlobal
            )
        })
        .cloned();
    let ac_global_section = sections
        .iter()
        .find(|section| matches!(section.section_kind, FrameSectionKind::AcGlobal))
        .cloned();
    let dc_group_sections = sections
        .iter()
        .filter_map(|section| match section.section_kind {
            FrameSectionKind::DcGroup { group } => {
                dc_groups
                    .get(group)
                    .copied()
                    .map(|group| VarDctGroupSectionMetadata {
                        section: section.clone(),
                        group,
                    })
            }
            _ => None,
        })
        .collect();
    let ac_group_sections = sections
        .iter()
        .filter_map(|section| match section.section_kind {
            FrameSectionKind::AcGroup { pass, group } => {
                ac_groups
                    .get(group)
                    .copied()
                    .map(|group| VarDctPassGroupSectionMetadata {
                        section: section.clone(),
                        pass,
                        group,
                    })
            }
            _ => None,
        })
        .collect();
    VarDctSectionBuckets {
        is_combined: sections
            .iter()
            .any(|section| matches!(section.section_kind, FrameSectionKind::Combined)),
        global_section,
        ac_global_section,
        ac_group_sections,
        dc_group_sections,
    }
}

fn group_metadata(
    groups_x: u32,
    groups_y: u32,
    group_dim: u32,
    frame_width: u32,
    frame_height: u32,
) -> Vec<VarDctGroupMetadata> {
    let mut groups = Vec::with_capacity(groups_x as usize * groups_y as usize);
    for gy in 0..groups_y {
        for gx in 0..groups_x {
            let x = gx * group_dim;
            let y = gy * group_dim;
            groups.push(VarDctGroupMetadata {
                group: groups.len(),
                x,
                y,
                width: group_dim.min(frame_width.saturating_sub(x)),
                height: group_dim.min(frame_height.saturating_sub(y)),
            });
        }
    }
    groups
}

fn group_intersects_region(group: &VarDctGroupMetadata, region: ImageRegion) -> bool {
    let Some(group_right) = group.x.checked_add(group.width) else {
        return true;
    };
    let Some(group_bottom) = group.y.checked_add(group.height) else {
        return true;
    };
    let Some(region_right) = region.x.checked_add(region.width) else {
        return true;
    };
    let Some(region_bottom) = region.y.checked_add(region.height) else {
        return true;
    };
    group.x < region_right
        && region.x < group_right
        && group.y < region_bottom
        && region.y < group_bottom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_multi_section_vardct_sections() {
        let ac_groups = vec![group(0, 0, 0, 128, 128), group(1, 128, 0, 128, 128)];
        let dc_groups = vec![group(0, 0, 0, 256, 128)];
        let sections = vec![
            section(0, 0, FrameSectionKind::DcGlobal),
            section(1, 1, FrameSectionKind::DcGroup { group: 0 }),
            section(2, 2, FrameSectionKind::AcGlobal),
            section(3, 3, FrameSectionKind::AcGroup { pass: 0, group: 0 }),
            section(4, 4, FrameSectionKind::AcGroup { pass: 0, group: 1 }),
        ];

        let buckets = classify_vardct_sections(&sections, &ac_groups, &dc_groups);

        assert!(!buckets.is_combined);
        assert_eq!(
            buckets.global_section.as_ref().unwrap().section_kind,
            FrameSectionKind::DcGlobal
        );
        assert_eq!(
            buckets.ac_global_section.as_ref().unwrap().section_kind,
            FrameSectionKind::AcGlobal
        );
        assert_eq!(buckets.dc_group_sections.len(), 1);
        assert_eq!(buckets.dc_group_sections[0].group, dc_groups[0]);
        assert_eq!(buckets.ac_group_sections.len(), 2);
        assert_eq!(buckets.ac_group_sections[0].pass, 0);
        assert_eq!(buckets.ac_group_sections[1].group, ac_groups[1]);
    }

    #[test]
    fn selects_group_sections_for_region() {
        let metadata = VarDctFrameMetadata {
            width: 256,
            height: 128,
            group_dim: 128,
            groups_x: 2,
            groups_y: 1,
            dc_groups_x: 1,
            dc_groups_y: 1,
            is_combined: false,
            global_section: Some(section(0, 0, FrameSectionKind::DcGlobal)),
            ac_global_section: Some(section(2, 2, FrameSectionKind::AcGlobal)),
            sections: Vec::new(),
            ac_groups: vec![group(0, 0, 0, 128, 128), group(1, 128, 0, 128, 128)],
            dc_groups: vec![group(0, 0, 0, 256, 128)],
            ac_group_sections: vec![
                VarDctPassGroupSectionMetadata {
                    section: section(3, 3, FrameSectionKind::AcGroup { pass: 0, group: 0 }),
                    pass: 0,
                    group: group(0, 0, 0, 128, 128),
                },
                VarDctPassGroupSectionMetadata {
                    section: section(4, 4, FrameSectionKind::AcGroup { pass: 0, group: 1 }),
                    pass: 0,
                    group: group(1, 128, 0, 128, 128),
                },
            ],
            dc_group_sections: vec![VarDctGroupSectionMetadata {
                section: section(1, 1, FrameSectionKind::DcGroup { group: 0 }),
                group: group(0, 0, 0, 256, 128),
            }],
        };

        let region = ImageRegion {
            x: 140,
            y: 8,
            width: 16,
            height: 16,
        };

        assert_eq!(
            metadata.ac_sections_for_region(region)[0]
                .section
                .section_logical_id,
            4
        );
        assert_eq!(
            metadata.dc_sections_for_region(region)[0]
                .section
                .section_logical_id,
            1
        );
    }

    fn section(
        logical_id: usize,
        physical_index: usize,
        kind: FrameSectionKind,
    ) -> VarDctSectionMetadata {
        VarDctSectionMetadata {
            section_logical_id: logical_id,
            section_physical_index: physical_index,
            section_kind: kind,
            codestream_offset: 100 + physical_index,
            payload_size: 10 + physical_index as u32,
        }
    }

    fn group(group: usize, x: u32, y: u32, width: u32, height: u32) -> VarDctGroupMetadata {
        VarDctGroupMetadata {
            group,
            x,
            y,
            width,
            height,
        }
    }
}
